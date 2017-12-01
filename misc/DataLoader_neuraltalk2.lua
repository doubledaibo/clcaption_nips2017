require 'hdf5'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local t = require 'misc.transforms'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
	
	-- load the json file which contains additional information about the dataset
	print('DataLoader loading json file: ', opt.json_file)
	self.info = utils.read_json(opt.json_file)
	self.ix_to_word = self.info.ix_to_word
	self.vocab_size = utils.count_keys(self.ix_to_word)

	self.batch_size = utils.getopt(opt, 'batch_size', 5) -- how many images get returned at one time (to go through CNN)
	self.seq_per_img = utils.getopt(opt, 'seq_per_img', 5) -- number of sequences to return per image

	print('vocab size is ' .. self.vocab_size)
	
	-- open the hdf5 file
	print('DataLoader loading h5 file: ', opt.h5_file)
	self.h5_file = hdf5.open(opt.h5_file, 'r')
 
	-- extract image size from dataset
	local images_size = self.h5_file:read('/images'):dataspaceSize()
	assert(#images_size == 4, '/images should be a 4D tensor')
	assert(images_size[3] == images_size[4], 'width and height must match')
	self.num_images = images_size[1]
	self.num_channels = images_size[2]
	self.max_image_size = images_size[3]

	print(string.format('read %d images of size %dx%dx%d', self.num_images, 
						self.num_channels, self.max_image_size, self.max_image_size))

	-- load in the sequence data
	local seq_size = self.h5_file:read('/labels'):dataspaceSize()
	self.seq_length = seq_size[2]
	print('max sequence length in data is ' .. self.seq_length)
	-- load the pointers in full to RAM (should be small enough)
	self.label_start_ix = self.h5_file:read('/label_start_ix'):all()
	self.label_end_ix = self.h5_file:read('/label_end_ix'):all()
	self.labels = self.h5_file:read('/labels'):all()
	self.label_lens = self.h5_file:read('/label_length'):all()
	-- separate out indexes for each of the provided splits
	self.split_ix = {}
	self.iterator = {}
	self.mmiterator = {}
	self.image_ids = {}
	self.mmperm = {}
	for i,img in pairs(self.info.images) do
		local split = img.split
		if not self.split_ix[split] then
			-- initialize new split
			self.split_ix[split] = {}
			self.iterator[split] = 1
			self.mmiterator[split] = 1
		end
		table.insert(self.split_ix[split], i)
		self.image_ids[i] = img.id		
	end
end

function DataLoader:init_rand(split)
	local size = #self.split_ix[split]	
	if split == 'train' then
		self.perm = torch.randperm(size)
	else
		self.perm = torch.range(1,size) -- for test and validation, do not permutate
	end
end

function DataLoader:reset_iterator(split)
	self.iterator[split] = 1
end

function DataLoader:reset_mmiterator(split)
	local size = #self.split_ix[split]
	self.mmperm[split] = torch.randperm(size)
	self.mmiterator[split] = 1
end

function DataLoader:getVocabSize()
	return self.vocab_size
end

function DataLoader:getVocab()
	return self.ix_to_word
end

function DataLoader:getSeqLength()
	return self.seq_length
end

function DataLoader:getnBatch(split)
	return math.floor(#self.split_ix[split] / self.batch_size)
end

function DataLoader:findMismatch(opt, data)
	local split = utils.getopt(opt, "split")
	if self.mmperm[split] == nil then
		self:reset_mmiterator(split)
	end
	local batch_size = self.batch_size
	local seq_per_img, seq_length = self.seq_per_img, self.seq_length
	local mmlabel_batch = torch.LongTensor(batch_size * seq_per_img, seq_length):zero()
	for i = 1, batch_size do
		local img_id = data.img_id[(i - 1) * seq_per_img + 1]
		while true do
			if self.mmiterator[split] > #self.split_ix[split] then
				self:reset_mmiterator(split)
			end
			local ix = self.split_ix[split][self.mmperm[split][self.mmiterator[split]]]
			self.mmiterator[split] = self.mmiterator[split] + 1
			if self.image_ids[ix] ~= img_id then
				local ix1 = self.label_start_ix[ix]
				local ix2 = self.label_end_ix[ix]
				local ixl = torch.random(ix1, ix2 - seq_per_img + 1)
				seq = self.labels[{{ixl, ixl + seq_per_img - 1}, {1, seq_length}}]
				ixl = (i - 1) * seq_per_img + 1
				mmlabel_batch[{{ixl, ixl + seq_per_img - 1}, {1, seq_length}}] = seq
				break
			end
		end	
	end	
	data.mmlabels = mmlabel_batch:transpose(1, 2):contiguous()	
end	

function DataLoader:run(opt)
	local split = utils.getopt(opt, 'split') -- lets require that user passes this in, for safety
	local size_image_use = utils.getopt(opt, 'size_image_use', -1)
	local size, batch_size = #self.split_ix[split], self.batch_size
	local seq_per_img, seq_length = self.seq_per_img, self.seq_length
	local num_channels, max_image_size = self.num_channels, self.max_image_size

	if size_image_use ~= -1 and size_image_use <= size then size = size_image_use end
	local split_ix = self.split_ix[split]
	local idx = self.iterator[split]
	
	if idx <= size then

		local indices = self.perm:narrow(1, idx, math.min(batch_size, size - idx + 1))

		local img_batch_raw = torch.ByteTensor(batch_size, 3, 256, 256)
		local label_batch = torch.LongTensor(batch_size * seq_per_img, seq_length):zero()
		local img_id_batch = {}
		for i, ixm in ipairs(indices:totable()) do
							
			local ix = split_ix[ixm]
			img_batch_raw[i] = self.h5_file:read("/images"):partial({ix, ix}, {1, 3}, {1, 256}, {1, 256})
			
			-- fetch the sequence labels
			local ix1 = self.label_start_ix[ix]
			local ix2 = self.label_end_ix[ix]						
			
			local ncap = ix2 - ix1 + 1 -- number of captions available for this image
			assert(ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t')
			local seq
			if ncap < seq_per_img then
				-- we need to subsample (with replacement)
				seq = torch.LongTensor(seq_per_img, seq_length)
				for q=1, seq_per_img do
					local ixl = torch.random(ix1,ix2)
					seq[{{q,q}}] = self.labels[{{ixl, ixl}, {1,seq_length}}]
				end
			else
				-- there is enough data to read a contiguous chunk, but subsample the chunk position
				local ixl = torch.random(ix1, ix2 - seq_per_img + 1) -- generates integer in the range
				seq = self.labels[{{ixl, ixl+seq_per_img-1}, {1,seq_length}}]
			end

			local il = (i-1)*seq_per_img+1
			label_batch[{{il,il+seq_per_img-1} }] = seq
			for k = il, il + seq_per_img - 1 do
				img_id_batch[k] = self.image_ids[ix]
			end
		end

		local data_augment = false
		if split == 'train' then
			data_augment = true
		end

		local batch_data = {}
		batch_data.labels = label_batch:transpose(1,2):contiguous()
		batch_data.images = self:prepro(img_batch_raw, data_augment)
		batch_data.img_id = img_id_batch
		
		self.iterator[split] = self.iterator[split] + batch_size
		return batch_data
	end
end

function DataLoader:prepro(imgs, data_augment)
	assert(data_augment ~= nil, 'pass this in. careful here.')

	local h,w = imgs:size(3), imgs:size(4)
	local cnn_input_size = 224

	-- cropping data augmentation, if needed
	if h > cnn_input_size or w > cnn_input_size then 
		local xoff, yoff
		if data_augment then
			xoff, yoff = torch.random(w-cnn_input_size), torch.random(h-cnn_input_size)
		else
			-- sample the center
			xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
		end
		-- crop.
		imgs = imgs[{ {}, {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]
	end

	-- ship to gpu or convert from byte to float
	imgs = imgs:float()

	-- lazily instantiate vgg_mean
	if not net_utils.vgg_mean then
		net_utils.vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1) -- in RGB order
	end
	net_utils.vgg_mean = net_utils.vgg_mean:typeAs(imgs) -- a noop if the types match

	-- subtract vgg mean
	imgs:add(-1, net_utils.vgg_mean:expandAs(imgs))

	return imgs
end

