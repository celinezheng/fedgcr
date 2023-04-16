import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import networks

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.num_classes = num_classes

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x, domain=None):
        raise NotImplementedError
    
class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.global_iter = 0
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        for param in self.featurizer.parameters():
            param.requires_grad = False
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        print(self.featurizer.n_outputs, self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        if self.hparams['vit_base_16']:
            optimizer = torch.optim.AdamW
        else:
            optimizer = torch.optim.Adam
        self.fulltune = False

        if self.hparams["lr"] > 0 and self.fulltune:
            # full finetune
            self.optimizer = optimizer(
                [
                    {'params': self.featurizer.parameters(), 'lr': self.hparams["lr"]},
                    {'params': self.classifier.parameters(), 'lr': self.hparams["lr_classifier"], 'weight_decay': 1e-5}
                ],
                weight_decay=self.hparams['weight_decay']
            )
        else:
            self.optimizer = optimizer(
                [
                    {'params': self.classifier.parameters(), 'lr': self.hparams["lr_classifier"]}
                ],
                weight_decay=1e-5,
            )
        
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x, domain=None):
        return self.network(x)

class PrependPrompt():
    def __init__(self, network, domain_token):
        self.featurizer = network
        self.domain_tokens = domain_token
        self.pos_embeddings = self.featurizer.network.encoder.pos_embedding
        
    def add_domain_prompt(self):
        domain_tokens = self.domain_tokens
        def _add_domain_prompt(model, x):
            act = x[0]
            x_new = torch.cat([act, domain_tokens], dim=1)
            return (x_new, )
        return _add_domain_prompt

    def __enter__(self):
        self.hook = self.featurizer.network.encoder.dropout.register_forward_pre_hook(self.add_domain_prompt())
        
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

class DoPrompt(ERM):
    def __init__(self, input_shape=(3, 224, 224), num_classes=10, num_domains=4, hparams=None):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains
        self.hidden_dim = self.featurizer.network.hidden_dim
        # self.prompt_dim = self.hparams['prompt_dim']
        self.prompt_dim = 8
        self.mlp_dim = 3072
        assert self.hparams['vit_base_16'] == True
        
        # prompt tokens
        self.prompt_tokens = nn.Parameter(
            torch.empty(num_domains, self.prompt_dim, self.featurizer.network.hidden_dim).normal_(std=0.02)
        )
        # prompt adapter
        self.project = networks.Project(self.hidden_dim, num_domains * self.prompt_dim, self.mlp_dim)
        
        # optimizer
        self.prompt_opt = torch.optim.AdamW(
            [self.prompt_tokens],
            lr=self.hparams["lr_prompt"],
            weight_decay=1e-5
        )
        self.project_opt = torch.optim.AdamW(
            self.project.parameters(),
            lr=self.hparams["lr_project"],
            weight_decay=self.hparams["wd_project"],
        )
        
        self.optimizer = torch.optim.AdamW(
            [
                # {'params': self.featurizer.parameters(), 'lr': self.hparams["lr"], 'weight_decay': self.hparams['weight_decay']},
                {'params': self.classifier.parameters(), 'lr': self.hparams["lr_classifier"], 'weight_decay': self.hparams['wd_classifier']}
            ]
        )
        
    def x_domain_prompt(self, x, domain):
        domain_labels = torch.full((len(x), ), domain, dtype=torch.int64, device="cuda")
        domain_tokens = self.prompt_tokens[domain_labels]
        return domain_tokens
        
    def x_domain_prompt_comb(self, all_bias):
        prompt_tokens = self.prompt_tokens
        tokens = prompt_tokens[None, ...]
        bias = all_bias[..., None]
        comb_prompt = tokens * bias
        comb_prompt = comb_prompt.sum(dim=1)
        return comb_prompt
    
    def minibatch_domain_prompt(self, client_idx, batch_size):
        domain_labels = torch.full((batch_size, ), client_idx, dtype=torch.int64, device="cuda")
        domain_tokens = self.prompt_tokens[domain_labels]
        return domain_tokens
    
    def all_bias_gt(self, disc_labels, all_bias):
        gt = torch.zeros(all_bias.shape[:-1], dtype=torch.float32, device="cuda")
        gt.scatter_(1, disc_labels.unsqueeze(1), 1)
        gt = gt.unsqueeze(-1).repeat(1, 1, self.prompt_dim)
        return gt
    
    
    @torch.no_grad()
    def forward_pk_feature(self, all_x, client_idx):
        domain_prompts = self.minibatch_domain_prompt(client_idx, all_x.shape[0])
        with PrependPrompt(self.featurizer, domain_prompts):
            feature, _ = self.featurizer.forward_vit_feature(all_x)
            # cls = self.featurizer(all_x)
            
        return feature, domain_prompts
    
    # todo
    def forward_pa_feature(self, all_x, all_z):
        hint = all_z.detach()
        all_bias = self.project(hint).reshape(-1, self.num_domains, self.prompt_dim)
        all_bias = F.softmax(all_bias, dim=1)
        domain_prompts = self.x_domain_prompt_comb(all_bias)
        with PrependPrompt(self.featurizer, domain_prompts):
            feature, cls = self.featurizer.forward_vit_feature(all_x)
            logits = self.classifier(cls)

        return logits, domain_prompts, feature

    def forward_prompt(self, all_x, client_idx):
        domain_prompts = self.minibatch_domain_prompt(client_idx, all_x.shape[0])
        with PrependPrompt(self.featurizer, domain_prompts):
            all_logit = self.network(all_x)
        return all_logit
    
    def forward_second(self, all_x, all_z):
        hint = all_z.detach()
        all_bias = self.project(hint).reshape(-1, self.num_domains, self.prompt_dim)
        all_bias = F.softmax(all_bias, dim=1)
        domain_prompts = self.x_domain_prompt_comb(all_bias)
        with PrependPrompt(self.featurizer, domain_prompts):
            all_logit = self.network(all_x)
        return all_logit, all_bias
    
    @torch.no_grad()
    def forward_raw(self, all_x):
        all_z = self.featurizer(all_x)
        return all_z
        
    def update_doprompt(self, all_x, all_y, client_idx, device, unlabeled=None):
        self.prompt_opt.zero_grad()
        self.optimizer.zero_grad()
        self.project_opt.zero_grad()
        
        
        # domain prompt learning
        all_logit = self.forward_prompt(all_x, client_idx)
        loss_dp = F.cross_entropy(all_logit, all_y)
        loss_dp.backward()

        # prompt adapter learning
        self.network.eval()
        all_z = self.forward_raw(all_x)
        self.network.train()
        
        disc_labels = torch.full((all_x.shape[0], ), client_idx, dtype=torch.int64, device=device)
        all_logit, all_bias = self.forward_second(all_x, all_z)
        gt_bias = self.all_bias_gt(disc_labels, all_bias)
        loss_w = F.binary_cross_entropy(all_bias, gt_bias)
        loss_a = F.cross_entropy(all_logit, all_y)
        (loss_a + self.hparams['lambda'] * loss_w).backward()
        
        pred = all_logit.data.max(1)[1]
        correct = pred.eq(all_y.view(-1)).sum().item()
        self.prompt_opt.step()
        self.optimizer.step()
        self.project_opt.step()

        return {
            "loss_dp": loss_dp.item(),  
            "loss_a": loss_a.item(),
            "loss_w": loss_w.item(),
            "loss": (loss_a + self.hparams['lambda'] * loss_w).item(),
            "correct": correct
        }
    
    def update(self, all_x, all_y, client_idx, device, unlabeled=None):
        self.prompt_opt.zero_grad()
        self.optimizer.zero_grad()
        self.project_opt.zero_grad()
        
        # disc_labels = torch.full((all_x.shape[0], ), client_idx, dtype=torch.int64, device=device)
        
        # domain prompt learning
        all_logit = self.forward_prompt(all_x, client_idx)
        loss_dp = F.cross_entropy(all_logit, all_y)
        loss_dp.backward()
        self.prompt_opt.step()

        # prompt adapter learning
        self.network.eval()
        all_z = self.forward_raw(all_x)
        # teacher_logit = self.forward_prompt(all_x, client_idx)
        # cls represent patch importance
        self.network.train()
        
        feat_k, pk = self.forward_pk_feature(all_x, client_idx)
        all_logit, pa, feat_a = self.forward_pa_feature(all_x, all_z)
        loss_sim = F.mse_loss(feat_a, feat_k)
        # loss_a = loss_fn_kd(all_logit, all_y, teacher_logit)

        loss_a = F.cross_entropy(all_logit, all_y)
        (loss_a + self.hparams['lambda'] * loss_sim).backward()
        # (loss_a + 0.1 * loss_sim).backward()
        
        pred = all_logit.data.max(1)[1]
        correct = pred.eq(all_y.view(-1)).sum().item()
        
        self.optimizer.step()
        self.project_opt.step()

        return {
            "loss_dp": loss_dp.item(),  
            "loss_a": loss_a.item(),
            "loss_sim": loss_sim.item(),
            "loss": (loss_a + self.hparams['lambda'] * loss_sim).item(),
            "correct": correct
        }
    
    def forward(self, x, domain=None):
        all_z = self.forward_raw(x)
        all_logit, _ = self.forward_second(x, all_z)
        return all_logit
    
    def predict(self, x, domain=None):
        all_z = self.forward_raw(x)
        all_logit, _ = self.forward_second(x, all_z)
        return all_logit

class FedPrompt(ERM):
    def __init__(self, input_shape=(3, 224, 224), num_classes=10, num_domains=1, hparams=None, lambda_con=0.5):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.hidden_dim = self.featurizer.network.hidden_dim
        self.prompt_num = 4
        self.mlp_dim = 3072
        self.lambda_con = lambda_con
        assert self.hparams['vit_base_16'] == True
        
        # prompt tokens
        self.prompt_tokens = nn.Parameter(
            torch.empty(self.prompt_num, self.featurizer.network.hidden_dim).normal_(std=0.02)
        )

        # image projector, similar to meta-net in CoCoOP
        self.project = networks.MetaNet(self.hidden_dim, self.prompt_num, self.hidden_dim, self.mlp_dim)
        
        # optimizer
        self.prompt_opt = torch.optim.AdamW(
            [self.prompt_tokens],
            lr=self.hparams["lr_prompt"],
            weight_decay=1e-5
        )

        self.project_opt = torch.optim.AdamW(
            self.project.parameters(),
            lr=self.hparams["lr_project"],
            weight_decay=self.hparams["wd_project"],
        )
        
        self.optimizer = torch.optim.AdamW(
            [
                # {'params': self.featurizer.parameters(), 'lr': self.hparams["lr"], 'weight_decay': self.hparams['weight_decay']},
                {'params': self.classifier.parameters(), 'lr': self.hparams["lr_classifier"], 'weight_decay': self.hparams['wd_classifier']}
            ]
        )
        
    def forward_prompt(self, x):
        repeat_prompt = self.prompt_tokens.repeat((x.shape[0], 1, 1)).to(self.prompt_tokens.device)
        with PrependPrompt(self.featurizer, repeat_prompt):
            logit = self.network(x)
        return logit
    
    @torch.no_grad()
    def forward_raw(self, all_x):
        all_z = self.featurizer(all_x)
        return all_z
        
    def forward_proj(self, x, z):
        img_proj = self.project(z)
        sample_prompt = img_proj.reshape((x.shape[0], self.prompt_num, self.featurizer.network.hidden_dim)).cuda()
        pi_repeat = self.prompt_tokens.repeat((x.shape[0], 1, 1)).to(self.prompt_tokens.device)
        comb_prompt = torch.concat((sample_prompt, pi_repeat), dim=1)
        with PrependPrompt(self.featurizer, comb_prompt):
            logit = self.network(x)
        return img_proj, logit
    
    def update(self, x, y, prompt_bank, gidx, device):
        self.prompt_opt.zero_grad()
        self.optimizer.zero_grad()
        self.project_opt.zero_grad()
        
        # domain prompt learning
        all_logit = self.forward_prompt(x)
        loss_p = F.cross_entropy(all_logit, y)
        loss_p.backward()
        self.prompt_opt.step()

        # prompt adapter learning
        self.network.eval()
        hint = self.forward_raw(x)
        self.network.train()
        # todo with gmap
        img_proj, logit = self.forward_proj(x, hint)
        loss_m = F.cross_entropy(logit, y)
        # server as new prompt for classification
        
        #if gidx == -1:
        reshape_pb = torch.concat((self.prompt_tokens.detach().unsqueeze(0), prompt_bank))
        labels = torch.zeros(x.shape[0], dtype=torch.long).to(device)
        # else:
        #     reshape_pb = prompt_bank
        #     # reshape_pb[gidx].data.copy_(self.prompt_tokens.detach())
        #     labels = torch.zeros(x.shape[0], dtype=torch.long).fill_(gidx).to(device)
        
        
        reshape_pb = torch.reshape(reshape_pb, (reshape_pb.shape[0], self.hidden_dim*self.prompt_num))
        anchor_dot_contrast = torch.matmul(reshape_pb, img_proj.T).to(device)

        # print(labels.shape, anchor_dot_contrast.shape)
        loss_con = F.cross_entropy(anchor_dot_contrast.T, labels)
        (self.lambda_con * loss_con + loss_m).backward()
        pred = all_logit.data.max(1)[1]
        correct = pred.eq(y.view(-1)).sum().item()

        self.optimizer.step()
        self.project_opt.step()

        return {
            "loss": (loss_p + loss_con).item(),
            "correct": correct
        }
    
    @torch.no_grad()
    def forward(self, x, prompt_bank):
        return self.forward_bank_sample(x, prompt_bank)
        # all_logit = self.forward_prompt(x)
        # return all_logit
    
    def predict(self, x):
        all_logit = self.forward_prompt(x)
        return all_logit
    
    def forward_bank_sample(self, x, prompt_bank):
        hint = self.forward_raw(x)
        img_proj = self.project(hint)
        sample_prompt = img_proj.reshape((x.shape[0], self.prompt_num, self.featurizer.network.hidden_dim)).cuda()
        reshape_pb = torch.reshape(prompt_bank, (prompt_bank.shape[0], self.hidden_dim*self.prompt_num))
        dot_contrast = F.softmax((torch.matmul(reshape_pb, img_proj.T).cuda()).T, dim=1)
        k = min(prompt_bank.shape[0], 3)
        domain_prods, domain_idxs = torch.topk(dot_contrast, k=k, dim=1)
        domain_prods = F.normalize(domain_prods, dim=1, p=1.0)
        domain_prods = domain_prods.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 4, 768)
        # Hadamard product
        chosen_tokens = domain_prods * prompt_bank[domain_idxs]
        chosen_tokens = torch.sum(chosen_tokens, dim=1)
        # combine domain prompt and sample prompt
        comb_prompt = torch.concat((sample_prompt, chosen_tokens), dim=1)
        with PrependPrompt(self.featurizer, comb_prompt):
            logit = self.network(x)
        return logit

    # todo : utilize the similarity of between input and prompt to weighted sum
    def forward_bank(self, x, prompt_bank):
        hint = self.forward_raw(x)
        img_proj = self.project(hint)
        reshape_pb = torch.reshape(prompt_bank, (prompt_bank.shape[0], self.hidden_dim*self.prompt_num))
        dot_contrast = F.softmax((torch.matmul(reshape_pb, img_proj.T).cuda()).T, dim=1)
        domain_prods, domain_idxs = torch.topk(dot_contrast, k=3, dim=1)
        domain_prods = F.normalize(domain_prods, dim=1, p=1.0)
        # chosen_tokens = torch.einsum('bcde,bc->bcde', prompt_bank[domain_idxs], domain_prods)
        domain_prods = domain_prods.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 4, 768)
        # Hadamard product
        chosen_tokens = domain_prods * prompt_bank[domain_idxs]
        chosen_tokens = torch.sum(chosen_tokens, dim=1)
        with PrependPrompt(self.featurizer, chosen_tokens):
            logit = self.network(x)
        return logit

class CoCoOP(ERM):
    def __init__(self, input_shape=(3, 224, 224), num_classes=10, hparams=None):
        super().__init__(input_shape, num_classes, 1, hparams)
        self.hidden_dim = self.featurizer.network.hidden_dim
        self.prompt_num = 4
        self.mlp_dim = 3072
        assert self.hparams['vit_base_16'] == True
        
        # prompt tokens
        self.prompt_tokens = nn.Parameter(
            torch.empty(self.prompt_num, self.featurizer.network.hidden_dim).normal_(std=0.02)
        )

        # image projector, similar to meta-net in CoCoOP
        self.meta_net = networks.MetaNet(self.hidden_dim, 1, self.hidden_dim, self.mlp_dim)
        
        # optimizer
        self.prompt_opt = torch.optim.AdamW(
            [self.prompt_tokens],
            lr=self.hparams["lr_prompt"],
            weight_decay=1e-5
        )

        self.project_opt = torch.optim.AdamW(
            self.meta_net.parameters(),
            lr=self.hparams["lr_project"],
            weight_decay=self.hparams["wd_project"],
        )
        
        self.optimizer = torch.optim.AdamW(
            [
                # {'params': self.featurizer.parameters(), 'lr': self.hparams["lr"], 'weight_decay': self.hparams['weight_decay']},
                {'params': self.classifier.parameters(), 'lr': self.hparams["lr_classifier"], 'weight_decay': self.hparams['wd_classifier']}
            ]
        )
        
    def forward_prompt(self, x):
        repeat_prompt = self.prompt_tokens.repeat((x.shape[0], 1, 1)).to(self.prompt_tokens.device)
        with PrependPrompt(self.featurizer, repeat_prompt):
            logit = self.network(x)
        return logit
    
    @torch.no_grad()
    def forward_raw(self, all_x):
        all_z = self.featurizer(all_x)
        return all_z
        
    def forward_proj(self, x, z):
        img_proj = self.meta_net(z)
        img_proj = img_proj.repeat((self.prompt_num, 1))
        img_proj = img_proj.reshape((x.shape[0], self.prompt_num, self.featurizer.network.hidden_dim)).cuda()
        pi_repeat = self.prompt_tokens.repeat((x.shape[0], 1, 1)).to(self.prompt_tokens.device)
        # comb_prompt = torch.concat((img_proj, pi_repeat), dim=1)
        comb_prompt = img_proj + pi_repeat
        with PrependPrompt(self.featurizer, comb_prompt):
            logit = self.network(x)
        return logit
    
    def update(self, x, y):
        self.prompt_opt.zero_grad()
        self.optimizer.zero_grad()
        self.project_opt.zero_grad()
        
        # domain prompt learning
        all_logit = self.forward_prompt(x)
        loss_p = F.cross_entropy(all_logit, y)
        loss_p.backward()
        self.prompt_opt.step()

        # prompt adapter learning
        self.network.eval()
        hint = self.forward_raw(x)
        self.network.train()
        # todo with gmap
        logit = self.forward_proj(x, hint)
        loss_m = F.cross_entropy(logit, y)      
        loss_m.backward()
        pred = all_logit.data.max(1)[1]
        correct = pred.eq(y.view(-1)).sum().item()

        self.optimizer.step()
        self.project_opt.step()

        return {
            "loss": (loss_p + loss_m).item(),
            "correct": correct
        }
    
    def forward_feat(self, x):
        hint = self.forward_raw(x)
        img_proj = self.meta_net(hint)
        img_proj = img_proj.repeat((self.prompt_num, 1))
        img_proj = img_proj.reshape((x.shape[0], self.prompt_num, self.featurizer.network.hidden_dim)).cuda()
        global_prompt = self.prompt_tokens.repeat((x.shape[0], 1, 1)).to(self.prompt_tokens.device)
        # combine domain prompt and sample prompt
        comb_prompt = global_prompt + img_proj
        with PrependPrompt(self.featurizer, comb_prompt):
            feat = self.featurizer(x)
        return feat

    def forward(self, x):
        return self.forward_meta(x)
    
    def predict(self, x):
        all_logit = self.forward_prompt(x)
        return all_logit
    
    def forward_meta(self, x):
        hint = self.forward_raw(x)
        img_proj = self.meta_net(hint)
        img_proj = img_proj.repeat((self.prompt_num, 1))
        img_proj = img_proj.reshape((x.shape[0], self.prompt_num, self.featurizer.network.hidden_dim)).cuda()
        global_prompt = self.prompt_tokens.repeat((x.shape[0], 1, 1)).to(self.prompt_tokens.device)
        # combine domain prompt and sample prompt
        comb_prompt = global_prompt + img_proj
        with PrependPrompt(self.featurizer, comb_prompt):
            logit = self.network(x)
        return logit

