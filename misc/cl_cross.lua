function cl_forward(split, opt, loader, protos, noise_protos, cl_crit, data, expanded_feats)
 -- preparation of noise model q
	local noise_feats_conv_fix = noise_protos.cnn_conv_fix:forward(data.aa_images)
	local noise_feats_conv = noise_protos.cnn_conv:forward(noise_feats_conv_fix)
	local noise_feats_conv_t = noise_protos.transform_cnn_conv:forward(noise_feats_conv)
	local noise_feats_fc = noise_protos.cnn_fc:forward(noise_feats_conv)
	
	local noise_expanded_feats_conv = noise_protos.expanderConv:forward(noise_feats_conv_t)
	local noise_expanded_feats_fc = noise_protos.expanderFC:forward(noise_feats_fc)
 -- preparation of target model
	local loss = 0
	local total_d_expanded_feats
 -- data samples
	local log_prob = protos.lm:forward({expanded_feats, data.labels[{{1, 16}, {}}]})
	local log_noise_prob = noise_protos.lm:forward({noise_expanded_feats_conv, noise_expanded_feats_fc, data.labels})
	loss = loss + cl_crit:forward({log_prob, log_noise_prob, data.labels, 1, 1.0})
	local d_log_probs = cl_crit:backward({})
	local d_expanded_feats, dummy = unpack(protos.lm:backward({}, d_log_probs))
	total_d_expanded_feats = d_expanded_feats:clone()
 -- noise samples
	for i = 1, opt.vu do
		loader:findMismatch({split = split}, data)
		data.mmlabels = data.mmlabels:cuda()
		log_prob = protos.lm:forward({expanded_feats, data.mmlabels[{{1, 16}, {}}]})
		log_noise_prob = noise_protos.lm:forward({noise_expanded_feats_conv, noise_expanded_feats_fc, data.mmlabels})
		loss = loss + cl_crit:forward({log_prob, log_noise_prob, data.mmlabels, 0, 1.0 / opt.vu})
		d_log_probs = cl_crit:backward({})
		d_expanded_feats, dummy = unpack(protos.lm:backward({}, d_log_probs))
		total_d_expanded_feats:add(d_expanded_feats)	
	end
	return loss, total_d_expanded_feats	
end


require 'nngraph'
require 'misc.OneHot'
require 'misc.Clip'
local crit, parent = torch.class('nn.CLCriterion', 'nn.Criterion')
function crit:__build_net(vocab_size, log_vu, is_data)
	local inputs = {}
	table.insert(inputs, nn.Identity()())  --seq
	table.insert(inputs, nn.Identity()())  --mask
	table.insert(inputs, nn.Identity()())  --log(p_m)
	table.insert(inputs, nn.Identity()())  --log(p_n)
	local seq = inputs[1]
	local mask = inputs[2]
	local logpm = inputs[3]
	local logpn = inputs[4]
	local onehot_seq = nn.OneHot(vocab_size + 1)(seq)
	local logpm_seq_unmasked = nn.Sum(2, 2)(nn.CMulTable()({logpm, onehot_seq}))
	local logpm_seq_masked = nn.Sum(1, 2)(nn.CMulTable()({logpm_seq_unmasked, mask}))
	
	local logpn_seq_unmasked = nn.Sum(2, 2)(nn.CMulTable()({logpn, onehot_seq}))
	local logpn_seq_masked = nn.Sum(1, 2)(nn.CMulTable()({logpn_seq_unmasked, mask}))
	
	local g_seq = nn.CSubTable()({logpm_seq_masked, logpn_seq_masked})
	local h_seq = nn.Sigmoid()(nn.AddConstant(-log_vu)(g_seq))
	local loss
	if is_data then
		loss = nn.Log()(nn.Clip(1e-20, 1.0)(h_seq))
	else
		loss = nn.Log()(nn.Clip(1e-20, 1.0)(nn.AddConstant(1)(nn.MulConstant(-1)(h_seq))))
	end
	local outputs = {}
	table.insert(outputs, loss)
	return nn.gModule(inputs, outputs)
end
function crit:__init(vocab_size, log_vu)
	parent.__init(self)
	self.data_net = self:__build_net(vocab_size, log_vu, true)
	self.noise_net = self:__build_net(vocab_size, log_vu, false)
	self.mask = torch.Tensor()
	self.grad = torch.Tensor()
	self.aug_seq = torch.Tensor()
	self.aug_logpm = torch.Tensor()
	self.grad_logpm = torch.Tensor()
end

function crit:updateOutput(inputs)
	local log_pm = inputs[1]
	local log_pn = inputs[2]
	local seq = inputs[3]
	local seq_length, batch_size = seq:size(1), seq:size(2)
	local end_token_idx = log_pm:size(3)
	self.aug_seq:resize(seq_length + 1, batch_size):zero()
	self.aug_logpm:resize(seq_length + 1, batch_size, end_token_idx):zero()
	self.grad_logpm:resizeAs(log_pm):zero()

	for i = 1, batch_size do
		for j = seq_length, 1, -1 do
			if seq[j][i] ~= 0 then
				self.aug_seq[{{1, j}, {i, i}}] = seq[{{1, j}, {i, i}}]
				self.aug_seq[j + 1][i] = end_token_idx
				if j < 17 then
					self.aug_logpm[{{1, j}, {i, i}, {}}] = log_pm[{{2, j + 1}, {i, i}, {}}]
					self.aug_logpm[{{j + 3, j + 3}, {i, i}, {}}] = log_pm[{{j + 2, j + 2}, {i, i}, {}}]
				else
					self.aug_logpm[{{1, 16}, {i, i}, {}}] = log_pm[{{2, 17}, {i, i}, {}}]
					self.aug_logpm[{{19, 19}, {i, i}, {}}] = log_pm[{{18, 18}, {i, i}, {}}]
				end
				break
			end
		end
	end
	
	self.is_data = inputs[4] 
	local weight = 1.0
	if #inputs > 4 then
		weight = inputs[5]
	end
	self.mask:resizeAs(self.aug_seq):fill(1)
	self.mask[torch.eq(self.aug_seq, 0)] = 0
	self.aug_seq[torch.eq(self.aug_seq, 0)] = end_token_idx
	self.inputs = {self.aug_seq, self.mask, self.aug_logpm, log_pn}
	local net_out
	if self.is_data == 1 then
		net_out = self.data_net:forward(self.inputs)
	else
		net_out = self.noise_net:forward(self.inputs)
	end
	self.grad:resizeAs(net_out):fill(0)
	local loss = 0
	for i = 1, batch_size do
		loss = loss - net_out[i] * weight
		self.grad[i] = - weight / batch_size
	end
	return loss / batch_size
end

function crit:updateGradInput(inputs)
	local dummy1, dummy2, grad_pm, dummy3
	if self.is_data == 1 then
		dummy1, dummy2, grad_pm, dummy3 = unpack(self.data_net:backward(self.inputs, self.grad))
	else
		dummy1, dummy2, grad_pm, dummy3 = unpack(self.noise_net:backward(self.inputs, self.grad))
	end
	local batch_size = self.aug_seq:size(2)
	for i = 1, batch_size do
		for j = 19, 1, -1 do
			if self.aug_seq[j][i] ~= 0 then
				if j > 17 then
					self.grad_logpm[{{2, 17}, {i, i}, {}}] = grad_pm[{{1, 16}, {i, i}, {}}]
					self.grad_logpm[{{18, 18}, {i, i}, {}}] = grad_pm[{{19, 19}, {i, i}, {}}]
				else
					self.grad_logpm[{{2, j}, {i, i}, {}}] = grad_pm[{{1, j - 1}, {i, i}, {}}]
					self.grad_logpm[{{j + 1, j + 1}, {i, i}, {}}] = grad_pm[{{j + 2, j + 2}, {i, i}, {}}]
				end
				break
			end
		end
	end
	return self.grad_logpm
end 
