require 'csvigo'
require 'torch'
require 'acdc'
require 'nn'

csv_path = arg[1]
fun = arg[2]

csv = csvigo.load({path = csv_path, verbose = false, mode = "raw"})

-- Read in csv.
file = torch.DiskFile(csv_path,'r')
M = file:readObject()

-- Apply fun.
if fun == 'dct' then
  MM = acdc.DCT():forward(M)
elseif fun == 'idct' then
  MM = acdc.IDCT():forward(M)
elseif fun == 'fastacdc_size_4' then
  local module = acdc.FastACDC(4, {
    sign_init = false,
    rand_init = false,
  }):cuda()
  MM = module:forward(M:cuda()):float()
else
  MM = nil
end

csvigo.save{path=csv_path, data=MM:totable(), mode='raw', header=false}
