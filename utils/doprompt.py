import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

class GCR(ERM):
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
        # if num_classes<10:
        #     self.mlp_dim = self.hidden_dim // 16
        self.meta_net = networks.MetaNet(self.hidden_dim, 1, self.hidden_dim, self.mlp_dim)
        
        # optimizer
        self.prompt_opt = torch.optim.Adam(
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
        self.all_opt = torch.optim.AdamW(
            [
                # {'params': self.featurizer.parameters(), 'lr': self.hparams["lr"], 'weight_decay': self.hparams['weight_decay']},
                {'params': [self.prompt_tokens], 'lr': self.hparams["lr_prompt"], 'weight_decay': 1e-5},
                {'params': self.meta_net.parameters(), 'lr': self.hparams["lr_project"], 'weight_decay': self.hparams['lr_project']},
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

    @torch.no_grad()
    def get_dc(self, x):
        hint = self.forward_raw(x)
        pi = self.meta_net(hint)
        return pi
        
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

    def forward_pi_cst(self, x):
        self.network.eval()
        z = self.forward_raw(x)
        self.network.train()
        pi = self.meta_net(z)
        img_proj = pi.repeat((self.prompt_num, 1))
        img_proj = img_proj.reshape((x.shape[0], self.prompt_num, self.featurizer.network.hidden_dim)).cuda()
        pi_repeat = self.prompt_tokens.repeat((x.shape[0], 1, 1)).to(self.prompt_tokens.device)
        # comb_prompt = torch.concat((img_proj, pi_repeat), dim=1)
        comb_prompt = img_proj + pi_repeat
        with PrependPrompt(self.featurizer, comb_prompt):
            feat = self.featurizer(x)
        logit = self.classifier(feat)
        return logit, pi, feat

    @torch.no_grad()
    def forward_pre(self, x, pre_net):
        hint = self.forward_raw(x)
        pi = pre_net.meta_net(hint)
        img_proj = pi.repeat((self.prompt_num, 1))
        img_proj = img_proj.reshape((x.shape[0], self.prompt_num, self.featurizer.network.hidden_dim)).cuda()
        global_prompt = pre_net.prompt_tokens.repeat((x.shape[0], 1, 1)).cuda()
        # combine domain prompt and sample prompt
        comb_prompt = global_prompt + img_proj
        with PrependPrompt(self.featurizer, comb_prompt):
            feat = self.featurizer(x)
        return pi, feat

    def update(self, loss_fun, x, y):
        self.prompt_opt.zero_grad()
        self.optimizer.zero_grad()
        self.project_opt.zero_grad()
        
        # domain prompt learning
        all_logit = self.forward_prompt(x)
        loss_p = loss_fun(all_logit, y)
        loss_p.backward()
        # self.prompt_opt.step()

        # prompt adapter learning
        self.network.eval()
        hint = self.forward_raw(x)
        self.network.train()
        # todo with gmap
        logit = self.forward_proj(x, hint)
        loss_m = loss_fun(logit, y)      
        loss_m.backward()
        pred = all_logit.data.max(1)[1]
        correct = pred.eq(y.view(-1)).sum().item()

        # self.optimizer.step()
        # self.project_opt.step()
        self.all_opt.step()

        return {
            "loss": (loss_p + loss_m).item(),
            "correct": correct
        }
    def update_con(self, w_con, loss_fun, x, y, cluster_pis, pre_pi, gidx, mean_feat, pre_feat, iter_ratio):
        self.prompt_opt.zero_grad()
        self.optimizer.zero_grad()
        self.project_opt.zero_grad()
        
        # domain prompt learning
        all_logit, pi, feat = self.forward_pi_cst(x)
        loss_m = loss_fun(all_logit, y)    

        cos = torch.nn.CosineSimilarity(dim=-1)
        temperature = 0.5
        base = 0.5
        mu = 0.5 + base * iter_ratio
        # moon on outpot feauture to CLUSTER feature
        posi = cos(cluster_pis[gidx], pi)
        logits_con = posi.reshape(-1,1)
        for i in range(cluster_pis.shape[0]):
            if i==gidx: continue
            nega = cos(cluster_pis[i], pi)
            logits_con = torch.cat((logits_con, nega.reshape(-1,1)), dim=1)
        nega = cos(pre_pi, pi)
        logits_con = torch.cat((logits_con, nega.reshape(-1,1)), dim=1)
        logits_con /= temperature
        # y_con = torch.full((x.shape[0],), gidx).cuda().long()
        y_con = torch.full((x.shape[0],), 0).cuda().long()
        loss_con1 = mu * loss_fun(logits_con, y_con)
        # moon on outpot feauture to GLOBAL feature
        posi = cos(mean_feat, feat)
        logits_con = posi.reshape(-1,1)
        nega = cos(pre_feat, feat)
        logits_con = torch.cat((logits_con, nega.reshape(-1,1)), dim=1)
        y_con = torch.full((x.shape[0],), 0).cuda().long()
        loss_con2 = loss_fun(logits_con, y_con)
        loss_con = loss_con1 + loss_con2
        loss = loss_m + w_con*loss_con

        loss.backward()
        pred = all_logit.data.max(1)[1]
        correct = pred.eq(y.view(-1)).sum().item()

        self.all_opt.step()

        return {
            "loss": (loss).item(),
            "correct": correct
        }
    def update_clscon(self, w_con, loss_fun, x, y, cluster_pis, pre_pi, gidx, pre_glob, pre_local, iter_ratio):
        self.prompt_opt.zero_grad()
        self.optimizer.zero_grad()
        self.project_opt.zero_grad()
        
        # domain prompt learning
        all_logit, pi, feat = self.forward_pi_cst(x)
        loss_m = loss_fun(all_logit, y)    

        cos = torch.nn.CosineSimilarity(dim=-1)
        temperature = 0.5
        base = 0.5
        mu = 0.5 + base * iter_ratio
        # moon on outpot feauture to CLUSTER feature
        posi = cos(cluster_pis[gidx], pi)
        logits_con = posi.reshape(-1,1)
        for i in range(cluster_pis.shape[0]):
            if i==gidx: continue
            nega = cos(cluster_pis[i], pi)
            logits_con = torch.cat((logits_con, nega.reshape(-1,1)), dim=1)
        nega = cos(pre_pi, pi)
        logits_con = torch.cat((logits_con, nega.reshape(-1,1)), dim=1)
        logits_con /= temperature
        # y_con = torch.full((x.shape[0],), gidx).cuda().long()
        y_con = torch.full((x.shape[0],), 0).cuda().long()
        loss_con1 = mu * loss_fun(logits_con, y_con)
        # moon on outpot feauture to GLOBAL feature
        _, _, z_glob = pre_glob.forward_pi_cst(x)
        _, _, z_prev = pre_local.forward_pi_cst(x)
        posi = cos(z_glob, feat)
        logits_con = posi.reshape(-1,1)
        nega = cos(z_prev, feat)
        logits_con = torch.cat((logits_con, nega.reshape(-1,1)), dim=1)
        y_con = torch.full((x.shape[0],), 0).cuda().long()
        loss_con2 = 0.5 * loss_fun(logits_con, y_con)
        loss_con = loss_con1 + loss_con2
        loss = loss_m + w_con * loss_con

        loss.backward()
        pred = all_logit.data.max(1)[1]
        correct = pred.eq(y.view(-1)).sum().item()

        self.all_opt.step()

        return {
            "loss": (loss).item(),
            "correct": correct
        }
    def update_moon(self, loss_fun, x, y, pre_glob, pre_local, iter_ratio):
        self.prompt_opt.zero_grad()
        self.optimizer.zero_grad()
        self.project_opt.zero_grad()
        
        # domain prompt learning
        all_logit, pi, z = self.forward_pi_cst(x)
        loss_m = loss_fun(all_logit, y)    

        mu = 1
        temperature = 0.5
        cos = torch.nn.CosineSimilarity(dim=-1)
        # moon on outpot feauture to CLUSTER feature
        _, z_glob = self.forward_pre(x, pre_glob)
        posi = cos(z_glob, z)
        logits_con = posi.reshape(-1,1)
        _, z_prev = self.forward_pre(x, pre_local)
        nega = cos(z_prev, z)
        logits_con = torch.cat((logits_con, nega.reshape(-1,1)), dim=1)
        logits_con /= temperature
        y_con = torch.full((x.shape[0],), 0).cuda().long()
        loss_con = mu * loss_fun(logits_con, y_con)
        
        loss = loss_m + loss_con

        loss.backward()
        pred = all_logit.data.max(1)[1]
        correct = pred.eq(y.view(-1)).sum().item()

        self.all_opt.step()

        return {
            "loss": (loss).item(),
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

