from .basic_controller import BasicMAC
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.nn as nn
import numpy as np
import copy


class MlpProdDecoder(nn.Module):
    """Hadamard-product-based MLP link predictor."""

    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.net = nn.Sequential(
            nn.Linear(embedding_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, h, u, v, adj=None):
        h_u = h[u]
        h_v = h[v]
        return self.net(h_u * h_v)

    def predict(self, h, u, v, adj=None):
        forward_res = self.forward(h, u, v)
        res = th.cat([th.sigmoid(forward_res)], dim=-1)
        return res


# This multi-agent controller shares parameters between agents
class ROLANDMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

        self.prediction_head = nn.Sequential(
            nn.Linear(args.gnn_dim, args.gnn_dim),
            nn.ReLU(),
            nn.Linear(args.gnn_dim, input_shape),
        ).cuda()  # TODO test pretext task to predict future link as ROLAND DOES: 
           # suppose to have hard-attention / sparse adj matrix MlpProdDecoder(args.gnn_dim, args.gnn_dim)
        
        self.model_init = None
        self.prev_node_states = None  # no previous state on day 0.
        # {'node_states': [Tensor, Tensor], 'node_cells: [Tensor, Tensor]}

        self.auc_hist = list()
        self.mrr_hist = list()
        
        # after not updating the best model for `tol` epochs, stop.
        self.meta_method = "moving_average"  # TODO: args.meta_method
        self.meta_alpha = 0.2
        self.tol = 1  # TODO: args.tol

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def train_step(self, agent_inputs, optimizer):
        hidden_states = self.hidden_states.detach()

        h = self.agent.rpz_forward(self.prev_node_states, hidden_states)
        z = self.prediction_head(h)


        mse_loss = nn.MSELoss()
        loss = mse_loss(z, agent_inputs)
        optimizer.zero_grad()
        th.cuda.empty_cache()
        loss.backward()
        optimizer.step()
        return loss

    @th.no_grad()
    def evaluate_step(self, agent_inputs):
        h = self.agent.rpz_forward(self.prev_node_states, self.hidden_states)
        z = self.prediction_head(h)
        mse_loss = nn.MSELoss()
        loss = mse_loss(z, agent_inputs)
        return loss

    def fine_tune(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        # agent_inputs = agent_inputs.detach()

        # current task: t --> t+1.
        # (1) Evaluate model's performance on this task, at this time, the
        # model has seen no information on t+1, this evaluation is fair.
        # perf = self.evaluate_step(agent_inputs)
        #   TODO  writer.add_scalars('val' if i == 1 else 'test', perf, t)
        if self.prev_node_states is not None:

            # (2) Reveal the ground truth of task (t, t+1) and update the model
            # to prepare for the next task.
            params = self.agent.rpz_params() + list(self.prediction_head.parameters())
            optimizer = th.optim.Adam(params=params, lr=self.args.lr_roland)

            # best model's validation loss, training epochs, and state_dict.
            best_model = {'val_loss': np.inf, 'train_epoch': 0, 'state': None}
            # keep track of how long we have NOT update the best model.
            best_model_unchanged = 0

            # internal training loop (intra-snapshot cross-validation).
            # choose the best model using current validation set, prepare for
            # next task.

            if self.model_init is not None:
                # For meta-learning, start fine-tuning from the pre-computed
                # initialization weight.
                self.agent.load_state_dict(copy.deepcopy(self.model_init))

            for i in range(5):
                # Start with the un-trained model (i = 0), evaluate the model.
                val_loss = self.evaluate_step(agent_inputs)

                if val_loss < best_model['val_loss']:
                    # replace the best model with the current model.
                    best_model = {'val_loss': val_loss, 'train_epoch': i,
                                'state': copy.deepcopy(self.agent.state_dict())}
                    best_model_unchanged = 0
                else:
                    # the current best model has dominated for these epochs.
                    best_model_unchanged += 1

                if best_model_unchanged >= self.tol:
                    # If the best model has not been updated for a while, stop.
                    break
                else:
                    # Otherwise, keep training.
                    train_perf = self.train_step(agent_inputs, optimizer)
            #      TODO   writer.add_scalars('train', train_perf, t)
            # writer.add_scalar('internal_best_val', best_model['val_loss'], t)
            # writer.add_scalar('best epoch', best_model['train_epoch'], t)

            # (3) Actually perform the update on training set to get node_states
            # contains information up to time t.
            # Use the best model selected from intra-snapshot cross-validation.
            self.agent.load_state_dict(best_model['state'])

            # update meta-learning's initialization weights.
            if self.model_init is None:  # for the first task.
                self.model_init = copy.deepcopy(best_model['state'])
            else:  # for subsequent task, update init.
                if self.meta_method == 'moving_average':
                    new_weight = self.meta_alpha
                elif self.meta_method == 'online_mean':
                    new_weight = 1 / (t + 1)  # for t=1, the second item, 1/2.
                else:
                    raise ValueError(f'Invalid method: {self.meta_method}')

                # (1-new_weight)*model_init + new_weight*best_model.
                self.model_init = average_state_dict(self.model_init,
                                                best_model['state'],
                                                new_weight)

        self.prev_node_states = agent_inputs.detach()


    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden(batch_size, self.n_agents)
        self.prev_node_states = None

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac, _copy=False):
        if _copy:
            self.agent.load_state_dict(copy.deepcopy(other_mac.agent.state_dict()))
        else:    
            self.agent.load_state_dict(other_mac.agent.state_dict())
            self.model_init = None

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load(
                "{}/agent.th".format(path), map_location=lambda storage, loc: storage
            )
        )

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )

        inputs = th.cat(
            [x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1
        )  # shape: (batch*nagent, obs_dim)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape

@th.no_grad()
def average_state_dict(dict1: dict, dict2: dict, weight: float) -> dict:
    # Average two model.state_dict() objects.
    # out = (1-w)*dict1 + w*dict2
    assert 0 <= weight <= 1
    d1 = copy.deepcopy(dict1)
    d2 = copy.deepcopy(dict2)
    out = dict()
    for key in d1.keys():
        assert isinstance(d1[key], th.Tensor)
        param1 = d1[key].detach().clone()
        assert isinstance(d2[key], th.Tensor)
        param2 = d2[key].detach().clone()
        out[key] = (1 - weight) * param1 + weight * param2
    return out