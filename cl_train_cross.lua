require 'torch'
require 'nn'
require 'nngraph'
require 'misc.DataLoader_cross'

local utils = require 'misc.utils'
require 'adaptiveattention.adaptiveattention'
require 'neuraltalk2.neuraltalk2'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'
require 'misc.cl_cross'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train an Image Captioning model')
cmd:text()
cmd:text('Options')


-- Data input settings

cmd:option('-input_h5','/data/coco/cocotalk.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','/data/coco/cocotalk.json','path to the json file containing additional info and vocab')
cmd:option('-cnn_model','../image_model/resnet-152.t7','path to CNN model file containing the weights, Caffe format. Note this MUST be a VGGNet-16 right now.')
cmd:option('-vgg_proto', '../grammar-caption/vgg_deploy.prototxt', '')
cmd:option('-vgg_model', '../grammar-caption/vgg_final.caffemodel', '')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-noise_start_from', '', '')
cmd:option('-vu', 5, '')
cmd:option('-checkpoint_path', 'save/', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-startEpoch', 1, 'Max number of training epoch')


cmd:option('-dataset','eval_cocotalk','')

-- Model settings
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-num_layers',1,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-input_encoding_size',512,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-batch_size',20,'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')

-- training setting
cmd:option('-nEpochs', 20, 'Max number of training epoch')
cmd:option('-finetune_cnn_after', 21, 'After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')

--actuall batch size = gpu_num * batch_size

cmd:option('-fc_size',2048,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-conv_size',2048,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-seq_per_img',5,'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')

-- Optimization: General
cmd:option('-grad_clip',0.1,'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')

-- Optimization: for the Language Model
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',4e-4,'learning rate')
cmd:option('-learning_rate_decay_start', 20, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')

-- Optimization: for the CNN
cmd:option('-cnn_optim','adam','optimization to use for CNN')
cmd:option('-cnn_optim_alpha',0.8,'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta',0.999,'alpha for momentum of CNN')
cmd:option('-cnn_learning_rate',1e-5,'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0, 'L2 weight decay just for the CNN')
cmd:option('-finetune_start_layer', 6, 'finetune start layer. [1-10]')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', -1, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 5, 'how often to save a model checkpoint?')
cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '1', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
--torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
	require 'cutorch'
	require 'cunn'
	if opt.backend == 'cudnn' then require 'cudnn' end
	cutorch.manualSeed(opt.seed)
	cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_file = opt.input_h5, json_file = opt.input_json,  
				batch_size = opt.batch_size, seq_per_img = opt.seq_per_img}

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
-- create protos from scratch
-- intialize language model
local lmOpt = {}
lmOpt.vocab_size = loader:getVocabSize()
lmOpt.input_encoding_size = opt.input_encoding_size
lmOpt.rnn_size = opt.rnn_size
lmOpt.num_layers = opt.num_layers
lmOpt.dropout = opt.drop_prob_lm
lmOpt.seq_length = loader:getSeqLength()
lmOpt.batch_size = opt.batch_size * opt.seq_per_img
lmOpt.fc_size = opt.fc_size
lmOpt.conv_size = opt.conv_size

local noise_checkpoint
if opt.noise_start_from ~= '' then
	noise_checkpoint = torch.load(opt.noise_start_from)
end
local noise_protos = {}
noise_protos.lm = nn.AdaptiveAttention(lmOpt):cuda()
if opt.noise_start_from ~= '' then 
	print("load noise model from " .. opt.noise_start_from)
	noise_protos.cnn_conv_fix = noise_checkpoint.protos.cnn_conv_fix:cuda()
	noise_protos.cnn_conv = noise_checkpoint.protos.cnn_conv:cuda()
	noise_protos.cnn_fc = noise_checkpoint.protos.cnn_fc:cuda()
else
	local cnn_raw = torch.load(opt.cnn_model)

	noise_protos.cnn_conv_fix = net_utils.build_residual_cnn_conv_fix(cnn_raw, {backend = cnn_backend, start_layer_num = opt.finetune_start_layer}):cuda()
	noise_protos.cnn_conv = net_utils.build_residual_cnn_conv(cnn_raw, {backend = cnn_backend, start_layer_num = opt.finetune_start_layer}):cuda()
	noise_protos.cnn_fc = net_utils.build_residual_cnn_fc(cnn_raw, {backend = cnn_backend}):cuda()
end
noise_protos.expanderConv = nn.FeatExpanderConv(opt.seq_per_img):cuda()
noise_protos.expanderFC = nn.FeatExpander(opt.seq_per_img):cuda()
noise_protos.transform_cnn_conv = net_utils.transform_cnn_conv(opt.conv_size):cuda()
noise_params, grad_noise_params = noise_protos.lm:getParameters()
if opt.noise_start_from ~= '' then
	noise_params:copy(noise_checkpoint.lmparam)
end
noise_protos.lm:createClones()
noise_protos.cnn_conv:evaluate()
noise_protos.cnn_fc:evaluate()
noise_protos.lm:evaluate()
noise_protos.cnn_conv_fix:evaluate()


local cl_crit = nn.CLCriterion(lmOpt.vocab_size, 0):cuda()

local loaded_checkpoint
if opt.start_from ~= '' then -- just copy to gpu1 params
	loaded_checkpoint = torch.load(opt.start_from)
end
local protos = {}
-- initialize the ConvNet 
if opt.start_from ~= '' then -- just copy to gpu1 params
	print("load checkpoint from " .. opt.start_from)
	protos.lm = loaded_checkpoint.protos.lm
	local lm_modules = protos.lm:getModulesList()
	for k, v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end
	protos.lm = protos.lm:cuda()
	protos.cnn = loaded_checkpoint.protos.cnn
	net_utils.unsanitize_gradients(protos.cnn)
	protos.cnn = protos.cnn:cuda()
else
	require 'loadcaffe'
	-- following original neuraltalk2
	lmOpt.seq_length = 16
	protos.lm = nn.Neuraltalk2(lmOpt):cuda()
	local cnn_raw = loadcaffe.load(opt.vgg_proto, opt.vgg_model, "cudnn")
	protos.cnn = net_utils.build_cnn(cnn_raw, {encoding_size = opt.input_encoding_size, backend = "cudnn"}):cuda()
end
protos.expanderFC = nn.FeatExpander(opt.seq_per_img):cuda()

params, grad_params = protos.lm:getParameters()
cnn1_params, cnn1_grad_params = protos.cnn:getParameters()

print('total number of parameters in LM: ', params:nElement())
print('total number of parameters in CNN_conv: ', cnn1_params:nElement())

assert(params:nElement() == grad_params:nElement())
assert(cnn1_params:nElement() == cnn1_grad_params:nElement())

protos.thin_lm = protos.lm:clone()                                                                                                                                                                             
protos.thin_lm.core:share(protos.lm.core, 'weight', 'bias')                                                                                                                                                    
protos.thin_lm.lookup_table:share(protos.lm.lookup_table, 'weight', 'bias')                                                                                                                                    
protos.thin_cnn = protos.cnn:clone('weight', 'bias')             

protos.lm:createClones()
collectgarbage() 

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function evaluate_split(split, evalopt)
	local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

	print('=> evaluating ...')
	-- setting to the evaluation mode, use only the first gpu
	protos.cnn:evaluate()
	protos.lm:evaluate()

	local n = 0
	local loss_sum = 0
	local loss_evals = 0
	local predictions = {}
	local vocab = loader:getVocab()
	local imgId_cell = {}

	local nbatch = math.ceil(val_images_use / opt.batch_size)
	if val_images_use == -1 then
		nbatch = loader:getnBatch(split)
	end
--	nbatch = 10
	loader:init_rand(split)
	loader:reset_iterator(split)

	for n = 1, nbatch do
		local data = loader:run({split = split, size_image_use = val_images_use})
		-- convert the data to cuda
		data.aa_images = data.aa_images:cuda()
		data.nt_images = data.nt_images:cuda()
		data.labels = data.labels:cuda()

		-- forward the model to get loss
		local feats = protos.cnn:forward(data.nt_images)		

		local expanded_feats = protos.expanderFC:forward(feats)
		local loss, dexpanded_feats = cl_forward(split, opt, loader, protos, noise_protos, cl_crit, data, expanded_feats)

		loss_sum = loss_sum + loss
		loss_evals = loss_evals + 1
		-- forward the model to also get generated samples for each image
		local seq = protos.lm:sample(expanded_feats, {beam_size = 3})
		local sents = net_utils.decode_sequence(vocab, seq)
		for k=1,#sents do
			local img_id = data.img_id[k]
			local entry
			if imgId_cell[img_id] == nil then -- make sure there are one caption for each image.
				imgId_cell[img_id] = 1
				entry = {image_id = img_id, caption = sents[k]}
				table.insert(predictions, entry)
				if n == 1 then -- print the first batch
					print(string.format('image %s: %s', entry.image_id, entry.caption))
				end
			end
		end
	end
	local lang_stats
	if opt.language_eval == 1 then
		local sampleOpt = {beam_size = 3}		
		lang_stats = net_utils.language_eval(predictions, {id = opt.id, dataset = opt.dataset},sampleOpt)
	end

	return loss_sum/loss_evals, predictions, lang_stats
end

-------------------------------------------------------------------------------
-- train function
-------------------------------------------------------------------------------
local function Train(epoch)

	local size_image_use = -1
	print('=> Training epoch # ' .. epoch)
	print('lm_learning_rate: ' .. learning_rate 
					.. ' cnn_learning_rate: ' .. cnn_learning_rate)

	protos.cnn:training()
	protos.lm:training()

	local nbatch = math.ceil(size_image_use / opt.batch_size)
	if size_image_use == -1 then
		nbatch = loader:getnBatch('train')
	end
	local ave_loss = 0
	loader:init_rand('train')
	loader:reset_iterator('train')
	for n = 1, nbatch do
		grad_params:zero()

		-- setting the gradient of the CNN network
		if epoch >= opt.finetune_cnn_after and opt.finetune_cnn_after ~= -1 then
			cnn1_grad_params:zero()
		end

		local data = loader:run({split = 'train', size_image_use = size_image_use})
		-- convert the data to cuda
		data.aa_images = data.aa_images:cuda()
		data.nt_images = data.nt_images:cuda()
		data.labels = data.labels:cuda()

		-- we have to expand out image features, once for each sentence
		local feats = protos.cnn:forward(data.nt_images)
		local expanded_feats = protos.expanderFC:forward(feats)
		
		local loss, dexpanded_feats = cl_forward('train', opt, loader, protos, noise_protos, cl_crit, data, expanded_feats)

		grad_params:clamp(-opt.grad_clip, opt.grad_clip)
		
		if epoch >= opt.finetune_cnn_after and opt.finetune_cnn_after ~= -1 then
			-- backprop the CNN, but only if we are finetuning
			local dfeats = protos.expanderFC:backward(feats, dexpanded_feats)
			local dummy = protos.cnn:backward(data.nt_images, dfeats)

			-- apply L2 regularization
			if opt.cnn_weight_decay > 0 then
				cnn1_grad_params:add(opt.cnn_weight_decay, cnn1_params)
			end

			cnn1_grad_params:clamp(-opt.grad_clip, opt.grad_clip)
		end
		if n % 10 == 0 then
			print("epoch: " .. epoch .. ", iter: " .. n .. " / " .. nbatch .. ", " .. "loss: " .. loss)
		end
		-----------------------------------------------------------------------------
		ave_loss = ave_loss + loss
		-- update the parameters
		if opt.optim == 'rmsprop' then
			rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
		elseif opt.optim == 'adam' then
			adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
		elseif opt.optim == 'sgd' then
			sgd(params, grad_params, learning_rate)
		else
			error('bad option opt.optim')
		end

		if epoch >= opt.finetune_cnn_after and opt.finetune_cnn_after ~= -1 then
			if opt.cnn_optim == 'sgd' then
				sgd(cnn1_params, cnn1_grad_params, cnn1_learning_rate)
			elseif opt.cnn_optim == 'sgdm' then
				sgdm(cnn1_params, cnn1_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, cnn1_optim_state)
			elseif opt.cnn_optim == 'adam' then
				adam(cnn1_params, cnn1_grad_params, cnn_learning_rate, opt.cnn_optim_alpha, opt.cnn_optim_beta, opt.optim_epsilon, cnn1_optim_state)
			else
				error('bad option for opt.cnn_optim')
			end
		end
	end
	ave_loss = ave_loss / nbatch

	return ave_loss
end


paths.mkdir(opt.checkpoint_path)

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
optim_state = {}
cnn1_optim_state = {}
learning_rate = opt.learning_rate
cnn_learning_rate = opt.cnn_learning_rate

local startEpoch = opt.startEpoch
local loss0
local loss_history = {}
local val_lang_stats_history = {}
local val_loss_history = {}
local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.id)
--evaluate_split("val", {val_images_use = opt.val_images_use})
for epoch = startEpoch, opt.nEpochs do

	-- doing the learning rate decay
	--[[
	if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
		local frac = (epoch - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
		local decay_factor = math.pow(0.5, frac)
		learning_rate = learning_rate * decay_factor -- set the decayed rate
	end
	]]--
--	if epoch > opt.learning_rate_decay_start then
--		learning_rate = 1e-4
--	end

	local train_loss = Train(epoch)
	print('training loss for # ' .. epoch .. ' : ' .. train_loss)
		
	-- save the model.
	if epoch % opt.save_checkpoint_every == 0 then
		local val_loss, val_predictions, lang_stats = evaluate_split('val', {val_images_use = opt.val_images_use})	
		print('validation loss for # ' .. epoch .. ' : ' .. val_loss)

		loss_history[epoch] = train_loss
		val_loss_history[epoch] = val_loss

		if lang_stats then
			val_lang_stats_history[epoch] = lang_stats
		end

		local checkpoint = {}
		checkpoint.loss_history = loss_history
		checkpoint.val_loss_history = val_loss_history
		checkpoint.val_predictions = val_predictions -- save these too for CIDEr/METEOR/etc eval
		checkpoint.val_lang_stats_history = val_lang_stats_history

		utils.write_json(checkpoint_path .. '.json', checkpoint)
		print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

		local save_protos = {}                                                                                                                                                                         
                save_protos.cnn = protos.thin_cnn                                                                                                                                                              
                save_protos.lm = protos.thin_lm                                                                                                                                                                
                checkpoint.protos = save_protos                                                                                                                                                                
                checkpoint.vocab = loader:getVocab()                                                                                                                                                           
                torch.save(checkpoint_path .. '_' .. epoch .. '.t7', checkpoint)                                                                                                                               
                print('wrote checkpoint to ' .. checkpoint_path .. '_' .. epoch .. '.t7')              
	end

end
