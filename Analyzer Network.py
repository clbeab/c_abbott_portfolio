import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedKFold
import torch
from torch.nn import Linear, ReLU, Softmax, LayerNorm, RNN, TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
import gc

# Taking all the models and flattening the parameters into a single torch tensor
def preprocess(arg):
    for model in range(400):
        data = torch.load(f'models/{arg}/model_{model}',map_location=torch.device('cpu'))
        new_data = []
        tensor_count = []
        key_names = []
        for key in data:
                if key == 'atrribs':
                     data[key] = torch.mean(data[key],dim=0) # taking an average across all the nodes in the simulation since it varies
                row = torch.flatten(data[key])
                new_data.append(row)
                tensor_count.append(row.shape[0])
                key_names.append(key)
        new_tensor_count = []
        name_set = []
        seen = set()
        for c in range(len(key_names)):
            name = key_names[c].partition(".")[0]
            if name not in seen:
                    seen.add(name)
                    name_set.append(name)
        temp_count = 0
        for x in range(len(name_set)):
            for c in range(len(key_names)):
                if key_names[c].partition(".")[0] == name_set[x]:
                    temp_count += tensor_count[c]
            new_tensor_count.append(temp_count)
        new_tensor_count = torch.tensor(new_tensor_count)
        new_comb_data = torch.concat(new_data,dim=-1)
        torch.save(new_comb_data,f'models/{arg}/Model Tensors/tensor{model}')
        torch.save(new_tensor_count,'models/tensor_count') # Just keeps track of which part of the GNN the different parameters came from

# Loads all the model tensors and creates a single data source and target list
def data_prep():
    nested_list = []
    targets = []
    for x in range(400):
        nested_list.append(torch.load(f'models/EM/Model Tensors/tensor{x}'))
        targets.append(torch.tensor((1.0,0,0)))
        nested_list.append(torch.load(f'models/box/Model Tensors/tensor{x}'))
        targets.append(torch.tensor((0,1.0,0)))
        nested_list.append(torch.load(f'models/Diffusion/Model Tensors/tensor{x}'))
        targets.append(torch.tensor((0,0,1.0)))
    torch.save(nested_list,'models/nested_tensor_lists')
    torch.save(targets,'models/target_list')
    print(len(nested_list))
    print(len(targets))

class Analyzer_model (torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.atrib = Linear(16,10) #A
        self.u_layer = Linear(16,10) #U
        self.in_1 = Linear(5248,50) #EV
        self.in_2 = Linear(768,50) #EE
        self.encoder_layer = Linear(100,20)
        self.in_3 = Linear(51328,50) #PE1
        self.in_4 = Linear(16512,50) #PE2
        self.in_5 = Linear(16512,50) #PE3
        self.in_6 = Linear(16512,50) #PE4
        self.in_7 = Linear(34944,50) #PV1
        self.in_8 = Linear(16512,50) #PV2
        self.in_9 = Linear(16512,50) #PV3
        self.in_10 = Linear(16512,50) #PV4
        self.in_11 = Linear(34944,50) #PU1
        self.in_12 = Linear(16512,50) #PU2
        self.in_13 = Linear(16512,50) #PU3
        self.in_14 = Linear(2064,50) #PU4
        self.message_layer = Linear(600,40)
        self.in_15 = Linear(16512,50) #D1
        self.in_16 = Linear(16512,50) #D2
        self.in_17 = Linear(16512,50) #D3
        self.in_18 = Linear(387,50) #D4
        self.decoder_layer = Linear(200,20)
        self.layer_2 = Linear(100,10)
        self.output_layer = Linear(10,3)
        self.act = ReLU()
        self.sm = Softmax(dim=-1)
        self.list = torch.nn.ModuleList()
        self.norm = LayerNorm(normalized_shape=50)

    def forward(self,input_list,tensor_list):
        gpu = torch.device('cuda')
        tensor_list.to(gpu)
        input_list.to(gpu)
        data = self.list
        data = []
        data.append(self.atrib(input_list[:,:tensor_list[0]]))
        data.append(self.u_layer(input_list[:,tensor_list[0]:tensor_list[1]]))
        data.append(self.in_1(input_list[:,tensor_list[1]:tensor_list[2]]))
        data.append(self.in_2(input_list[:,tensor_list[2]:tensor_list[3]]))
        data.append(self.in_3(input_list[:,tensor_list[3]:tensor_list[4]]))
        data.append(self.in_4(input_list[:,tensor_list[4]:tensor_list[5]]))
        data.append(self.in_5(input_list[:,tensor_list[5]:tensor_list[6]]))
        data.append(self.in_6(input_list[:,tensor_list[6]:tensor_list[7]]))
        data.append(self.in_7(input_list[:,tensor_list[7]:tensor_list[8]]))
        data.append(self.in_8(input_list[:,tensor_list[8]:tensor_list[9]]))
        data.append(self.in_9(input_list[:,tensor_list[9]:tensor_list[10]]))
        data.append(self.in_10(input_list[:,tensor_list[10]:tensor_list[11]]))
        data.append(self.in_11(input_list[:,tensor_list[11]:tensor_list[12]]))
        data.append(self.in_12(input_list[:,tensor_list[12]:tensor_list[13]]))
        data.append(self.in_13(input_list[:,tensor_list[13]:tensor_list[14]]))
        data.append(self.in_14(input_list[:,tensor_list[14]:tensor_list[15]]))
        data.append(self.in_15(input_list[:,tensor_list[15]:tensor_list[16]]))
        data.append(self.in_16(input_list[:,tensor_list[16]:tensor_list[17]]))
        data.append(self.in_17(input_list[:,tensor_list[17]:tensor_list[18]]))
        data.append(self.in_18(input_list[:,tensor_list[18]:tensor_list[19]]))
        for x in range(len(data)):
            data[x] = self.act(data[x])
        data_e = (torch.cat(data[2:4],dim=-1))
        data_m = (torch.cat(data[4:16],dim=-1))
        data_d = (torch.cat(data[16:20],dim=-1))
        next_layer = []
        next_layer.append(data[0])
        next_layer.append(data[1])
        data_e = next_layer.append(self.encoder_layer(data_e))
        data_m = next_layer.append(self.message_layer(data_m))
        data_d = next_layer.append(self.decoder_layer(data_d))
        for x in range(len(next_layer)):
            next_layer[x] = self.act(next_layer[x])
        data = torch.cat(next_layer,dim=-1)
        data = self.layer_2(data)
        data = self.act(data)
        data = self.output_layer(data)
        data = self.sm(data)
        return data

    def name(self):
         return 'Analyzer'

class AnalyzerMini (torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.atrib = Linear(16,10) #A
        self.in_15 = Linear(16512,50) #D1
        self.in_16 = Linear(16512,50) #D2
        self.in_17 = Linear(16512,50) #D3
        self.in_18 = Linear(387,50) #D4
        self.decoder_layer = Linear(200,20)
        self.layer_2 = Linear(30,10)
        self.output_layer = Linear(10,3)
        self.act = ReLU()
        self.sm = Softmax(dim=-1)
        self.list = torch.nn.ModuleList()

    def forward(self,input_list,tensor_list):
        gpu = torch.device('cuda')
        tensor_list.to(gpu)
        input_list.to(gpu)
        data = self.list
        data = []
        data.append(self.atrib(input_list[:,:tensor_list[0]]))
        data.append(self.in_15(input_list[:,tensor_list[15]:tensor_list[16]]))
        data.append(self.in_16(input_list[:,tensor_list[16]:tensor_list[17]]))
        data.append(self.in_17(input_list[:,tensor_list[17]:tensor_list[18]]))
        data.append(self.in_18(input_list[:,tensor_list[18]:tensor_list[19]]))
        for x in range(len(data)):
            data[x] = self.act(data[x])
        data_d = torch.cat(data[1:5],dim=-1)
        next_layer = []
        data_e = next_layer.append(data[0])
        data_d = next_layer.append(self.decoder_layer(data_d))
        for x in range(len(next_layer)):
            next_layer[x] = self.act(next_layer[x])
        data = torch.cat(next_layer,dim=-1)
        data = self.layer_2(data)
        data = self.act(data)
        data = self.output_layer(data)
        data = self.sm(data)
        return data

    def name(self):
         return 'AnalyzerMini'

class Encoder(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs) 
        self.atrib = Linear(16,10) #A
        self.u_layer = Linear(16,10) #U
        self.in_1 = Linear(5248,500) #EV
        self.in_2 = Linear(768,500) #EE
        self.in_3 = Linear(51328,500) #PE1
        self.in_4 = Linear(16512,500) #PE2
        self.in_5 = Linear(16512,500) #PE3
        self.in_6 = Linear(16512,500) #PE4
        self.in_7 = Linear(34944,500) #PV1
        self.in_8 = Linear(16512,500) #PV2
        self.in_9 = Linear(16512,500) #PV3
        self.in_10 = Linear(16512,500) #PV4
        self.in_11 = Linear(34944,500) #PU1
        self.in_12 = Linear(16512,500) #PU2
        self.in_13 = Linear(16512,500) #PU3
        self.in_14 = Linear(2064,500) #PU4
        self.in_15 = Linear(16512,500) #D1
        self.in_16 = Linear(16512,500) #D2
        self.in_17 = Linear(16512,500) #D3
        self.in_18 = Linear(387,500) #D4
        self.latent = Linear(9020,4096)
        self.ratrib = Linear(10,16) #A
        self.ru_layer = Linear(10,16) #U
        self.rin_1 = Linear(500,5248) #EV
        self.rin_2 = Linear(500,768) #EE
        self.rin_3 = Linear(500,51328) #PE1
        self.rin_4 = Linear(500,16512) #PE2
        self.rin_5 = Linear(500,16512) #PE3
        self.rin_6 = Linear(500,16512) #PE4
        self.rin_7 = Linear(500,34944) #PV1
        self.rin_8 = Linear(500,16512) #PV2
        self.rin_9 = Linear(500,16512) #PV3
        self.rin_10 = Linear(500,16512) #PV4
        self.rin_11 = Linear(500,34944) #PU1
        self.rin_12 = Linear(500,16512) #PU2
        self.rin_13 = Linear(500,16512) #PU3
        self.rin_14 = Linear(500,2064) #PU4
        self.rin_15 = Linear(500,16512) #D1
        self.rin_16 = Linear(500,16512) #D2
        self.rin_17 = Linear(500,16512) #D3
        self.rin_18 = Linear(500,387) #D4
        self.rlatent = Linear(4096,9020)
        self.act = ReLU()

    def forward(self,input_list,tensor_list,decode=True):
        gpu = torch.device('cuda')
        tensor_list.to(gpu)
        input_list.to(gpu)
        data = []
        data.append(self.atrib(input_list[:,:tensor_list[0]]))
        data.append(self.u_layer(input_list[:,tensor_list[0]:tensor_list[1]]))
        data.append(self.in_1(input_list[:,tensor_list[1]:tensor_list[2]]))
        data.append(self.in_2(input_list[:,tensor_list[2]:tensor_list[3]]))
        data.append(self.in_3(input_list[:,tensor_list[3]:tensor_list[4]]))
        data.append(self.in_4(input_list[:,tensor_list[4]:tensor_list[5]]))
        data.append(self.in_5(input_list[:,tensor_list[5]:tensor_list[6]]))
        data.append(self.in_6(input_list[:,tensor_list[6]:tensor_list[7]]))
        data.append(self.in_7(input_list[:,tensor_list[7]:tensor_list[8]]))
        data.append(self.in_8(input_list[:,tensor_list[8]:tensor_list[9]]))
        data.append(self.in_9(input_list[:,tensor_list[9]:tensor_list[10]]))
        data.append(self.in_10(input_list[:,tensor_list[10]:tensor_list[11]]))
        data.append(self.in_11(input_list[:,tensor_list[11]:tensor_list[12]]))
        data.append(self.in_12(input_list[:,tensor_list[12]:tensor_list[13]]))
        data.append(self.in_13(input_list[:,tensor_list[13]:tensor_list[14]]))
        data.append(self.in_14(input_list[:,tensor_list[14]:tensor_list[15]]))
        data.append(self.in_15(input_list[:,tensor_list[15]:tensor_list[16]]))
        data.append(self.in_16(input_list[:,tensor_list[16]:tensor_list[17]]))
        data.append(self.in_17(input_list[:,tensor_list[17]:tensor_list[18]]))
        data.append(self.in_18(input_list[:,tensor_list[18]:tensor_list[19]]))
        for x in range(len(data)):
            data[x] = self.act(data[x])
        data_e = (torch.cat(data[2:4],dim=-1))
        data_m = (torch.cat(data[4:16],dim=-1))
        data_d = (torch.cat(data[16:20],dim=-1))
        next_layer = []
        next_layer.append(data[0])
        next_layer.append(data[1])
        next_layer.append(data_e)
        next_layer.append(data_m)
        next_layer.append(data_d)
        data = torch.cat(next_layer,dim=-1)
        data = self.latent(data)
        if not decode:
            return data
        else:
            data = self.rlatent(data)
            data = self.act(data)
            reverse_list = []
            reverse_list.append(self.ratrib(data[:,0:10]))
            reverse_list.append(self.ru_layer(data[:,10:20]))
            reverse_list.append(self.rin_1(data[:,20:520]))
            reverse_list.append(self.rin_2(data[:,520:1020]))
            reverse_list.append(self.rin_3(data[:,1020:1520]))
            reverse_list.append(self.rin_4(data[:,1520:2020]))
            reverse_list.append(self.rin_5(data[:,2020:2520]))
            reverse_list.append(self.rin_6(data[:,2520:3020]))
            reverse_list.append(self.rin_7(data[:,3020:3520]))
            reverse_list.append(self.rin_8(data[:,3520:4020]))
            reverse_list.append(self.rin_9(data[:,4020:4520]))
            reverse_list.append(self.rin_10(data[:,4520:5020]))
            reverse_list.append(self.rin_11(data[:,5020:5520]))
            reverse_list.append(self.rin_12(data[:,5520:6020]))
            reverse_list.append(self.rin_13(data[:,6020:6520]))
            reverse_list.append(self.rin_14(data[:,6520:7020]))
            reverse_list.append(self.rin_15(data[:,7020:7520]))
            reverse_list.append(self.rin_16(data[:,7520:8020]))
            reverse_list.append(self.rin_17(data[:,8020:8520]))
            reverse_list.append(self.rin_18(data[:,8520:9020]))
            data = torch.cat(reverse_list,dim=-1)
            return data
        
    def name(self):
        return 'Encoder'

class Encoder_PhysResNet(torch.nn.Module):
    def __init__(self):
        super(Encoder_PhysResNet, self).__init__()
        base_model = torchvision.models.resnet50()
        base_model.fc = Linear(2048,3)
        self.base_model = base_model
        self.encoder = Encoder()
        self.encoder.load_state_dict(torch.load('Encoder'))
        self.output_layer = Softmax(dim=-1)


    def forward(self,input_list,tensor_list):
        x = self.encoder.forward(input_list,tensor_list,decode=False)       
        x = torch.reshape(x,(x.shape[0],1,64,64))
        x = x.repeat((1,3,1,1))
        x = self.base_model(x)
        x = self.output_layer(x)
        return x

    def name(self):
         return 'Encoder_PhysResNet'
 
class PhysResNet(torch.nn.Module):
    def __init__(self):
        super(PhysResNet, self).__init__()
        self.input_layer = Linear(3620,1728)
        base_model = torchvision.models.resnet50()
        base_model.fc = Linear(2048,3)
        self.base_model = base_model
        self.atrib = Linear(16,10) #A
        self.u_layer = Linear(16,10) #U
        self.in_1 = Linear(5248,200) #EV
        self.in_2 = Linear(768,200) #EE
        self.in_3 = Linear(51328,200) #PE1
        self.in_4 = Linear(16512,200) #PE2
        self.in_5 = Linear(16512,200) #PE3
        self.in_6 = Linear(16512,200) #PE4
        self.in_7 = Linear(34944,200) #PV1
        self.in_8 = Linear(16512,200) #PV2
        self.in_9 = Linear(16512,200) #PV3
        self.in_10 = Linear(16512,200) #PV4
        self.in_11 = Linear(34944,200) #PU1
        self.in_12 = Linear(16512,200) #PU2
        self.in_13 = Linear(16512,200) #PU3
        self.in_14 = Linear(2064,200) #PU4
        self.in_15 = Linear(16512,200) #D1
        self.in_16 = Linear(16512,200) #D2
        self.in_17 = Linear(16512,200) #D3
        self.in_18 = Linear(387,200) #D4
        self.act = ReLU()
        self.output_layer = Softmax(dim=-1)
        self.list = torch.nn.ModuleList()

    def forward(self,input_list,tensor_list):
        data = self.list
        data = []
        data.append(self.atrib(input_list[:,:tensor_list[0]]))
        data.append(self.u_layer(input_list[:,tensor_list[0]:tensor_list[1]]))
        data.append(self.in_1(input_list[:,tensor_list[1]:tensor_list[2]]))
        data.append(self.in_2(input_list[:,tensor_list[2]:tensor_list[3]]))
        data.append(self.in_3(input_list[:,tensor_list[3]:tensor_list[4]]))
        data.append(self.in_4(input_list[:,tensor_list[4]:tensor_list[5]]))
        data.append(self.in_5(input_list[:,tensor_list[5]:tensor_list[6]]))
        data.append(self.in_6(input_list[:,tensor_list[6]:tensor_list[7]]))
        data.append(self.in_7(input_list[:,tensor_list[7]:tensor_list[8]]))
        data.append(self.in_8(input_list[:,tensor_list[8]:tensor_list[9]]))
        data.append(self.in_9(input_list[:,tensor_list[9]:tensor_list[10]]))
        data.append(self.in_10(input_list[:,tensor_list[10]:tensor_list[11]]))
        data.append(self.in_11(input_list[:,tensor_list[11]:tensor_list[12]]))
        data.append(self.in_12(input_list[:,tensor_list[12]:tensor_list[13]]))
        data.append(self.in_13(input_list[:,tensor_list[13]:tensor_list[14]]))
        data.append(self.in_14(input_list[:,tensor_list[14]:tensor_list[15]]))
        data.append(self.in_15(input_list[:,tensor_list[15]:tensor_list[16]]))
        data.append(self.in_16(input_list[:,tensor_list[16]:tensor_list[17]]))
        data.append(self.in_17(input_list[:,tensor_list[17]:tensor_list[18]]))
        data.append(self.in_18(input_list[:,tensor_list[18]:tensor_list[19]]))
        for x in range(len(data)):
            data[x] = self.act(data[x])
        x = torch.cat(data,dim=-1)
        x = self.input_layer(x)
        x = self.act(x)
        x = torch.reshape(x,(x.shape[0],3,24,24))
        x = self.base_model(x)
        x = self.output_layer(x)
        return x

    def name(self):
         return 'PhysResNet'
    
class PhysResNetFull(torch.nn.Module):
    def __init__(self):
        super(PhysResNetFull, self).__init__()
        self.output_layer = Softmax(dim=-1)
        base_model = torchvision.models.resnet50()
        base_model.fc = Linear(2048,3)
        self.base_model = base_model
        self.act = ReLU()
        self.list = torch.nn.ModuleList()

    def forward(self,input_list,tensor_list):
        x = input_list[:,:311052]
        # x = self.input_layer(x)
        # x = self.act(x)
        x = torch.reshape(x,(x.shape[0],3,322,322))
        x = self.base_model(x)
        x = self.output_layer(x)
        return x

    def name(self):
         return 'PhysResNet'

class Models_Dataset(Dataset):
    def __init__(self,nested_input_list,target_list):
        super().__init__()
        self.inputs = nested_input_list
        self.targets = target_list

    def __len__(self):
        return(len(self.inputs))

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

class PhysRNN(torch.nn.Module):
    def __init__(self,n):
        super().__init__()
        self.n = n
        self.atrib = Linear(16,n) #A
        self.u_layer = Linear(16,n) #U
        self.in_1 = Linear(5248,n) #EV
        self.in_2 = Linear(768,n) #EE
        self.in_3 = Linear(51328,n) #PE1
        self.in_4 = Linear(16512,n) #PE2
        self.in_5 = Linear(16512,n) #PE3
        self.in_6 = Linear(16512,n) #PE4
        self.in_7 = Linear(34944,n) #PV1
        self.in_8 = Linear(16512,n) #PV2
        self.in_9 = Linear(16512,n) #PV3
        self.in_10 = Linear(16512,n) #PV4
        self.in_11 = Linear(34944,n) #PU1
        self.in_12 = Linear(16512,n) #PU2
        self.in_13 = Linear(16512,n) #PU3
        self.in_14 = Linear(2064,n) #PU4
        self.in_15 = Linear(16512,n) #D1
        self.in_16 = Linear(16512,n) #D2
        self.in_17 = Linear(16512,n) #D3
        self.in_18 = Linear(387,n) #D4
        self.output_layer = Linear(n,3)
        self.RNN = RNN(n,n,1)
        self.RNN2 = RNN(n,n,1)
        self.act = ReLU()
        self.sm = Softmax(dim=-1)
        self.norm = LayerNorm(normalized_shape=n)

    def forward(self,input_list,tensor_list):
        h0 = torch.zeros((1,self.n))
        h1 = torch.zeros((1,self.n))
        gpu = torch.device('cuda')
        tensor_list.to(gpu)
        input_list.to(gpu)
        data = []
        data.append(self.atrib(input_list[:,:tensor_list[0]]))
        data.append(self.u_layer(input_list[:,tensor_list[0]:tensor_list[1]]))
        data.append(self.in_1(input_list[:,tensor_list[1]:tensor_list[2]]))
        data.append(self.in_2(input_list[:,tensor_list[2]:tensor_list[3]]))
        data.append(self.in_3(input_list[:,tensor_list[3]:tensor_list[4]]))
        data.append(self.in_4(input_list[:,tensor_list[4]:tensor_list[5]]))
        data.append(self.in_5(input_list[:,tensor_list[5]:tensor_list[6]]))
        data.append(self.in_6(input_list[:,tensor_list[6]:tensor_list[7]]))
        data.append(self.in_7(input_list[:,tensor_list[7]:tensor_list[8]]))
        data.append(self.in_8(input_list[:,tensor_list[8]:tensor_list[9]]))
        data.append(self.in_9(input_list[:,tensor_list[9]:tensor_list[10]]))
        data.append(self.in_10(input_list[:,tensor_list[10]:tensor_list[11]]))
        data.append(self.in_11(input_list[:,tensor_list[11]:tensor_list[12]]))
        data.append(self.in_12(input_list[:,tensor_list[12]:tensor_list[13]]))
        data.append(self.in_13(input_list[:,tensor_list[13]:tensor_list[14]]))
        data.append(self.in_14(input_list[:,tensor_list[14]:tensor_list[15]]))
        data.append(self.in_15(input_list[:,tensor_list[15]:tensor_list[16]]))
        data.append(self.in_16(input_list[:,tensor_list[16]:tensor_list[17]]))
        data.append(self.in_17(input_list[:,tensor_list[17]:tensor_list[18]]))
        data.append(self.in_18(input_list[:,tensor_list[18]:tensor_list[19]]))
        for x in range(len(data)):
            in_put = self.act(data[x])
            output1, (h0) = self.RNN(in_put,h0)
            output, (h1) = self.RNN(output1,h1)
        x = self.act(output)
        x = self.output_layer(x)
        x = self.sm(x)
        return x
    
    def name(self):
        return 'PhysRNN'

class MLP_RNN_voter_model (torch.nn.Module):
    def __init__(self,embedding_size):
        super().__init__()
        self.atrib = Linear(16,embedding_size) #A
        self.u_layer = Linear(16,embedding_size) #U
        self.in_1 = Linear(5248,embedding_size) #EV
        self.in_2 = Linear(768,embedding_size) #EE
        self.encoder_layer = Linear(embedding_size*2,20)
        self.in_3 = Linear(51328,embedding_size) #PE1
        self.in_4 = Linear(16512,embedding_size) #PE2
        self.in_5 = Linear(16512,embedding_size) #PE3
        self.in_6 = Linear(16512,embedding_size) #PE4
        self.in_7 = Linear(34944,embedding_size) #PV1
        self.in_8 = Linear(16512,embedding_size) #PV2
        self.in_9 = Linear(16512,embedding_size) #PV3
        self.in_10 = Linear(16512,embedding_size) #PV4
        self.in_11 = Linear(34944,embedding_size) #PU1
        self.in_12 = Linear(16512,embedding_size) #PU2
        self.in_13 = Linear(16512,embedding_size) #PU3
        self.in_14 = Linear(2064,embedding_size) #PU4
        self.message_layer = Linear(embedding_size*12,40)
        self.in_15 = Linear(16512,embedding_size) #D1
        self.in_16 = Linear(16512,embedding_size) #D2
        self.in_17 = Linear(16512,embedding_size) #D3
        self.in_18 = Linear(387,embedding_size) #D4
        self.decoder_layer = Linear(embedding_size*4,20)
        self.layer_2 = Linear(80+embedding_size*2,10)
        self.output_layer = Linear(10,3)
        self.RNN_output = Linear(embedding_size,3)
        self.act = ReLU()
        self.sm = Softmax(dim=-1)
        self.list = torch.nn.ModuleList()
        self.RNN = RNN(embedding_size,embedding_size,1)
        self.voter = Linear(6,3)
        self.embed_size = embedding_size


    def forward(self,input_list,tensor_list):
        gpu = torch.device('cuda')
        h0 = torch.zeros((1,self.embed_size))
        tensor_list.to(gpu)
        input_list.to(gpu)
        data = self.list
        data = []
        data.append(self.atrib(input_list[:,:tensor_list[0]]))
        data.append(self.u_layer(input_list[:,tensor_list[0]:tensor_list[1]]))
        data.append(self.in_1(input_list[:,tensor_list[1]:tensor_list[2]]))
        data.append(self.in_2(input_list[:,tensor_list[2]:tensor_list[3]]))
        data.append(self.in_3(input_list[:,tensor_list[3]:tensor_list[4]]))
        data.append(self.in_4(input_list[:,tensor_list[4]:tensor_list[5]]))
        data.append(self.in_5(input_list[:,tensor_list[5]:tensor_list[6]]))
        data.append(self.in_6(input_list[:,tensor_list[6]:tensor_list[7]]))
        data.append(self.in_7(input_list[:,tensor_list[7]:tensor_list[8]]))
        data.append(self.in_8(input_list[:,tensor_list[8]:tensor_list[9]]))
        data.append(self.in_9(input_list[:,tensor_list[9]:tensor_list[10]]))
        data.append(self.in_10(input_list[:,tensor_list[10]:tensor_list[11]]))
        data.append(self.in_11(input_list[:,tensor_list[11]:tensor_list[12]]))
        data.append(self.in_12(input_list[:,tensor_list[12]:tensor_list[13]]))
        data.append(self.in_13(input_list[:,tensor_list[13]:tensor_list[14]]))
        data.append(self.in_14(input_list[:,tensor_list[14]:tensor_list[15]]))
        data.append(self.in_15(input_list[:,tensor_list[15]:tensor_list[16]]))
        data.append(self.in_16(input_list[:,tensor_list[16]:tensor_list[17]]))
        data.append(self.in_17(input_list[:,tensor_list[17]:tensor_list[18]]))
        data.append(self.in_18(input_list[:,tensor_list[18]:tensor_list[19]]))
        for x in range(len(data)):
            in_put = self.act(data[x])
            output, (h0) = self.RNN(in_put,h0)
            data[x] = self.act(data[x])
        rnn = self.act(output)
        rnn = self.RNN_output(rnn)
        rnn = self.sm(rnn)
        data_e = (torch.cat(data[2:4],dim=-1))
        data_m = (torch.cat(data[4:16],dim=-1))
        data_d = (torch.cat(data[16:20],dim=-1))
        next_layer = []
        next_layer.append(data[0])
        next_layer.append(data[1])
        data_e = next_layer.append(self.encoder_layer(data_e))
        data_m = next_layer.append(self.message_layer(data_m))
        data_d = next_layer.append(self.decoder_layer(data_d))
        for x in range(len(next_layer)):
            next_layer[x] = self.act(next_layer[x])
        data = torch.cat(next_layer,dim=-1)
        data = self.layer_2(data)
        data = self.act(data)
        data = self.output_layer(data)
        data = self.sm(data)
        final = self.voter(torch.cat((data,rnn),dim=-1))
        final = self.sm(final)
        return final

    def name(self):
         return 'Analyzer'

class Transformer(torch.nn.Module):
    def __init__(self, embedding_size,nhead,num_encoder_layers) -> None:
        super().__init__()
        self.em_size = embedding_size
        encoder_layer = TransformerEncoderLayer(embedding_size,nhead,batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer,num_encoder_layers)
        self.atrib = Linear(16,embedding_size) #A
        self.u_layer = Linear(16,embedding_size) #U
        self.in_1 = Linear(5248,embedding_size) #EV
        self.in_2 = Linear(768,embedding_size) #EE
        self.in_3 = Linear(51328,embedding_size) #PE1
        self.in_4 = Linear(16512,embedding_size) #PE2
        self.in_5 = Linear(16512,embedding_size) #PE3
        self.in_6 = Linear(16512,embedding_size) #PE4
        self.in_7 = Linear(34944,embedding_size) #PV1
        self.in_8 = Linear(16512,embedding_size) #PV2
        self.in_9 = Linear(16512,embedding_size) #PV3
        self.in_10 = Linear(16512,embedding_size) #PV4
        self.in_11 = Linear(34944,embedding_size) #PU1
        self.in_12 = Linear(16512,embedding_size) #PU2
        self.in_13 = Linear(16512,embedding_size) #PU3
        self.in_14 = Linear(2064,embedding_size) #PU4
        self.in_15 = Linear(16512,embedding_size) #D1
        self.in_16 = Linear(16512,embedding_size) #D2
        self.in_17 = Linear(16512,embedding_size) #D3
        self.in_18 = Linear(387,embedding_size) #D4
        self.act = ReLU()
        self.output = Linear(20*embedding_size,3)
        self.sm = Softmax(dim=-1)

    def forward(self,input_list,tensor_list):
        batch_shape = (input_list.shape[0])
        gpu = torch.device('cuda')
        tensor_list.to(gpu)
        input_list.to(gpu)
        data = []
        data.append(self.atrib(input_list[:,:tensor_list[0]]))
        data.append(self.u_layer(input_list[:,tensor_list[0]:tensor_list[1]]))
        data.append(self.in_1(input_list[:,tensor_list[1]:tensor_list[2]]))
        data.append(self.in_2(input_list[:,tensor_list[2]:tensor_list[3]]))
        data.append(self.in_3(input_list[:,tensor_list[3]:tensor_list[4]]))
        data.append(self.in_4(input_list[:,tensor_list[4]:tensor_list[5]]))
        data.append(self.in_5(input_list[:,tensor_list[5]:tensor_list[6]]))
        data.append(self.in_6(input_list[:,tensor_list[6]:tensor_list[7]]))
        data.append(self.in_7(input_list[:,tensor_list[7]:tensor_list[8]]))
        data.append(self.in_8(input_list[:,tensor_list[8]:tensor_list[9]]))
        data.append(self.in_9(input_list[:,tensor_list[9]:tensor_list[10]]))
        data.append(self.in_10(input_list[:,tensor_list[10]:tensor_list[11]]))
        data.append(self.in_11(input_list[:,tensor_list[11]:tensor_list[12]]))
        data.append(self.in_12(input_list[:,tensor_list[12]:tensor_list[13]]))
        data.append(self.in_13(input_list[:,tensor_list[13]:tensor_list[14]]))
        data.append(self.in_14(input_list[:,tensor_list[14]:tensor_list[15]]))
        data.append(self.in_15(input_list[:,tensor_list[15]:tensor_list[16]]))
        data.append(self.in_16(input_list[:,tensor_list[16]:tensor_list[17]]))
        data.append(self.in_17(input_list[:,tensor_list[17]:tensor_list[18]]))
        data.append(self.in_18(input_list[:,tensor_list[18]:tensor_list[19]]))
        for x in range(len(data)):
            data[x] = self.act(data[x])
        data = torch.stack(data,dim=1)
        x = self.transformer(data)
        x = torch.reshape(x,(batch_shape,20*self.em_size))
        x = self.output(x)
        x = self.sm(x)
        return x

    def name(self):
         return 'Transformer'

class Transformer_RNN_Voter(torch.nn.Module):
    def __init__(self, embedding_size,nhead,num_encoder_layers) -> None:
        super().__init__()
        self.em_size = embedding_size
        encoder_layer = TransformerEncoderLayer(embedding_size,nhead,batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer,num_encoder_layers)
        self.atrib = Linear(16,embedding_size) #A
        self.u_layer = Linear(16,embedding_size) #U
        self.in_1 = Linear(5248,embedding_size) #EV
        self.in_2 = Linear(768,embedding_size) #EE
        self.in_3 = Linear(51328,embedding_size) #PE1
        self.in_4 = Linear(16512,embedding_size) #PE2
        self.in_5 = Linear(16512,embedding_size) #PE3
        self.in_6 = Linear(16512,embedding_size) #PE4
        self.in_7 = Linear(34944,embedding_size) #PV1
        self.in_8 = Linear(16512,embedding_size) #PV2
        self.in_9 = Linear(16512,embedding_size) #PV3
        self.in_10 = Linear(16512,embedding_size) #PV4
        self.in_11 = Linear(34944,embedding_size) #PU1
        self.in_12 = Linear(16512,embedding_size) #PU2
        self.in_13 = Linear(16512,embedding_size) #PU3
        self.in_14 = Linear(2064,embedding_size) #PU4
        self.in_15 = Linear(16512,embedding_size) #D1
        self.in_16 = Linear(16512,embedding_size) #D2
        self.in_17 = Linear(16512,embedding_size) #D3
        self.in_18 = Linear(387,embedding_size) #D4
        self.act = ReLU()
        self.output = Linear(20*embedding_size,3)
        self.sm = Softmax(dim=-1)
        self.RNN_output = Linear(embedding_size,3)
        self.RNN = RNN(embedding_size,embedding_size,1)
        self.voter = Linear(6,3)

    def forward(self,input_list,tensor_list):
        batch_shape = (input_list.shape[0])
        gpu = torch.device('cuda')
        h0 = torch.zeros((1,self.em_size))
        tensor_list.to(gpu)
        input_list.to(gpu)
        data = []
        data.append(self.atrib(input_list[:,:tensor_list[0]]))
        data.append(self.u_layer(input_list[:,tensor_list[0]:tensor_list[1]]))
        data.append(self.in_1(input_list[:,tensor_list[1]:tensor_list[2]]))
        data.append(self.in_2(input_list[:,tensor_list[2]:tensor_list[3]]))
        data.append(self.in_3(input_list[:,tensor_list[3]:tensor_list[4]]))
        data.append(self.in_4(input_list[:,tensor_list[4]:tensor_list[5]]))
        data.append(self.in_5(input_list[:,tensor_list[5]:tensor_list[6]]))
        data.append(self.in_6(input_list[:,tensor_list[6]:tensor_list[7]]))
        data.append(self.in_7(input_list[:,tensor_list[7]:tensor_list[8]]))
        data.append(self.in_8(input_list[:,tensor_list[8]:tensor_list[9]]))
        data.append(self.in_9(input_list[:,tensor_list[9]:tensor_list[10]]))
        data.append(self.in_10(input_list[:,tensor_list[10]:tensor_list[11]]))
        data.append(self.in_11(input_list[:,tensor_list[11]:tensor_list[12]]))
        data.append(self.in_12(input_list[:,tensor_list[12]:tensor_list[13]]))
        data.append(self.in_13(input_list[:,tensor_list[13]:tensor_list[14]]))
        data.append(self.in_14(input_list[:,tensor_list[14]:tensor_list[15]]))
        data.append(self.in_15(input_list[:,tensor_list[15]:tensor_list[16]]))
        data.append(self.in_16(input_list[:,tensor_list[16]:tensor_list[17]]))
        data.append(self.in_17(input_list[:,tensor_list[17]:tensor_list[18]]))
        data.append(self.in_18(input_list[:,tensor_list[18]:tensor_list[19]]))
        for x in range(len(data)):
            in_put = self.act(data[x])
            output, (h0) = self.RNN(in_put,h0)
            data[x] = self.act(data[x])
        rnn = self.act(output)
        rnn = self.RNN_output(rnn)
        rnn = self.sm(rnn)
        data = torch.stack(data,dim=1)
        x = self.transformer(data)
        x = torch.reshape(x,(batch_shape,20*self.em_size))
        x = self.output(x)
        x = self.sm(x)
        final = self.voter(torch.cat((x,rnn),dim=-1))
        final = self.sm(final)
        return final

    def name(self):
         return 'Transformer'

def initial_run(model_type, *kwargs):
    gpu = torch.device('cuda')

    # Load and Final Data prep
    data = torch.load('models/nested_tensor_lists')
    data = torch.stack(data)
    targets = torch.load('models/target_list')
    targets = torch.stack(targets)
    tar_id = torch.argmax(targets,dim=1)

    # Setup for training
    kfold = StratifiedKFold(n_splits=5)
    torch.set_default_device(gpu)
    k_acc = []
    confusion_matrix = np.zeros((3,3))

    # Training loop
    for k, (train, test) in enumerate(kfold.split(data,tar_id)):
        model = model_type(*kwargs)
        model.to(gpu)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,lambda epoch:0.99) # Don't actually use this anymore but I left it there for a print statement
        model_dataset_train = Models_Dataset(data[train],targets[train])
        model_dataset_test = Models_Dataset(data[test],targets[test])

        batch_size = 20
        dataloader = DataLoader(model_dataset_train,batch_size,False)
        test_dataloader = DataLoader(model_dataset_test,batch_size,False)

        tensor_list = torch.load('models/tensor_count')
        tensor_list = tensor_list.to(gpu)

        val_best = 100
        best_val_acc = 0
        val_not_improve_count = 0 # Was using this to implement early stopping but got rid of it.  Left it in as extra information.
        loss_list = []
        for epoch in range(200):
            model.train()
            progress_n = 20
            train_loss = 0
            for sample, target in dataloader:
                rand = torch.randn_like(sample)
                sample += (0.03*rand)
                sample = sample.to(gpu)
                target = target.to(gpu)
                result = model.forward(sample,tensor_list)
                loss = loss_fn(result,target)
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                # print(f'{progress_n}/960   ',end='\r')
                progress_n += 20
            correct = 0
            tot_count = 0
            model.eval()
            val_loss = 0
            for sample, target in test_dataloader:
                sample = sample.to(gpu)
                target = target.to(gpu)
                val_result = model.forward(sample,tensor_list)
                val_loss += loss_fn(val_result,target).item()
                for row in range(val_result.shape[0]):
                    if torch.argmax(val_result[row,:])==torch.argmax(target[row,:]):
                            correct += 1
                    tot_count += 1
            loss_list.append(val_loss)
            if val_loss < val_best:
                val_best = val_loss
                best_val_acc = correct/tot_count
                val_not_improve_count = 0
                torch.save(model.state_dict(),f'{model.name()} fold-{k}')
            elif epoch > 10:
                val_not_improve_count += 1
            if epoch == 10:
                val_best = 100
                model.load_state_dict(torch.load(f'{model.name()} fold-{k}'))
            print(f'k {k} : epoch {epoch} lr {scheduler.get_lr()}: acc {int(correct/tot_count*100)} : count {val_not_improve_count} : val loss {val_loss}  : train loss {train_loss}                                                         ',end='\r')
            if epoch >= 10:
                 optimizer.param_groups[0]['lr'] = 0.001 *(210-epoch)/200 # custom LR schedule
            if val_not_improve_count > 20:
                 break
            if correct/tot_count == 1:
                break
            gc.collect()
        k_acc.append(best_val_acc)

        # Loading the final trained model and getting results for this fold for the confusion matrix
        model.load_state_dict(torch.load(f'{model.name()} fold-{k}'))
        for sample, target in test_dataloader:
                sample = sample.to(gpu)
                target = target.to(gpu)
                confusion_result = model.forward(sample,tensor_list)
                for row in range(val_result.shape[0]):
                        confusion_matrix[torch.argmax(target[row,:]),torch.argmax(confusion_result[row,:])] +=1
        print('                                                                                                                                        ',end='\r')
        print(f'Fold {k} best acc {best_val_acc}')
        # print('Confusion Matrix:','\n',confusion_matrix)
        np.save(f'{model.name()} Losses {k}',loss_list)
    k_acc = np.array(k_acc)
    print(f'k fold average {np.average(k_acc)}')
    np.save(f'{model.name()} Confusion Matrix',confusion_matrix)
    print('Confusion Matrix:','\n',confusion_matrix)

def vector_run(model_type, *kwargs):
    gpu = torch.device('cuda')

    # Load and Final Data prep
    data = torch.load('models/nested_tensor_lists')
    data = torch.stack(data)
    targets = torch.load('models/target_list')
    targets = torch.stack(targets)
    new_targets = torch.zeros((targets.shape[0],4))
    for row in range(targets.shape[0]):
        if torch.argmax(targets[row,:]) == 0:
            new_targets[row,2] = 1
            new_targets[row,3] = 1
        elif torch.argmax(targets[row,:]) == 1:
            new_targets[row,0] = 1
            new_targets[row,3] = 1
        elif torch.argmax(targets[row,:]) == 2:
            new_targets[row,1] = 1
        else:
            print('not correct format')
    tar_id = torch.argmax(targets,dim=1)

    # Setup for training
    kfold = StratifiedKFold(n_splits=5)
    torch.set_default_device(gpu)
    k_acc = []
    confusion_matrix = np.zeros((4,4))

    # Training loop
    for k, (train, test) in enumerate(kfold.split(data,tar_id)):
        model = model_type(*kwargs)
        model.to(gpu)
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,lambda epoch:0.99) # Don't actually use this anymore but I left it there for a print statement
        model_dataset_train = Models_Dataset(data[train],new_targets[train])
        model_dataset_test = Models_Dataset(data[test],new_targets[test])

        batch_size = 20
        dataloader = DataLoader(model_dataset_train,batch_size,False)
        test_dataloader = DataLoader(model_dataset_test,batch_size,False)

        tensor_list = torch.load('models/tensor_count')
        tensor_list = tensor_list.to(gpu)

        val_best = 100
        best_val_acc = 0
        val_not_improve_count = 0 # Was using this to implement early stopping but got rid of it.  Left it in as extra information.
        loss_list = []
        for epoch in range(200):
            model.train()
            progress_n = 20
            train_loss = 0
            for sample, target in dataloader:
                rand = torch.randn_like(sample)
                sample += (0.03*rand)
                sample = sample.to(gpu)
                target = target.to(gpu)
                result = model.forward(sample,tensor_list)
                loss = loss_fn(result,target)
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                # print(f'{progress_n}/960   ',end='\r')
                progress_n += 20
            correct = 0
            tot_count = 0
            model.eval()
            val_loss = 0
            for sample, target in test_dataloader:
                sample = sample.to(gpu)
                target = target.to(gpu)
                val_result = model.forward(sample,tensor_list)
                val_loss += loss_fn(val_result,target).item()
                for row in range(val_result.shape[0]):
                    for v in range(4):
                        if (val_result[row,v]) > 0.5:
                            val_result[row,v] = 1
                        else:
                            val_result[row,v] = 0
                    if torch.equal(val_result[row,:] , target[row,: ]):
                        correct += 1
                    tot_count += 1
            loss_list.append(val_loss)
            if val_loss < val_best:
                val_best = val_loss
                best_val_acc = correct/tot_count
                val_not_improve_count = 0
                torch.save(model.state_dict(),f'{model.name()} fold-{k}')
            elif epoch > 10:
                val_not_improve_count += 1
            if epoch == 10:
                val_best = 100
                model.load_state_dict(torch.load(f'{model.name()} fold-{k}'))
            print(f'k {k} : epoch {epoch} lr {scheduler.get_lr()}: acc {int(correct/tot_count*100)} : count {val_not_improve_count} : val loss {val_loss}  : train loss {train_loss}                                                         ',end='\r')
            if epoch >= 10:
                 optimizer.param_groups[0]['lr'] = 0.001 *(210-epoch)/200 # custom LR schedule
            if val_not_improve_count > 20:
                 break
            if correct/tot_count == 1:
                break
            gc.collect()
        k_acc.append(best_val_acc)

        # Loading the final trained model and getting results for this fold for the confusion matrix
        model.load_state_dict(torch.load(f'{model.name()} fold-{k}'))
        for sample, target in test_dataloader:
                sample = sample.to(gpu)
                target = target.to(gpu)
                confusion_result = model.forward(sample,tensor_list)
                for row in range(val_result.shape[0]):
                        confusion_matrix[torch.argmax(target[row,:]),torch.argmax(confusion_result[row,:])] +=1
        print('                                                                                                                                        ',end='\r')
        print(f'Fold {k} best acc {best_val_acc}')
        # print('Confusion Matrix:','\n',confusion_matrix)
        np.save(f'{model.name()} Losses {k}',loss_list)
    k_acc = np.array(k_acc)
    print(f'k fold average {np.average(k_acc)}')
    np.save(f'{model.name()} Confusion Matrix',confusion_matrix)
    print('Confusion Matrix:','\n',confusion_matrix)

def encoder_train():
    gpu = torch.device('cuda') # Was running on GPU initially but this actually runs better on my CPU

    # Load and Final Data prep
    data = torch.load('models/nested_tensor_lists')
    data = torch.stack(data)
    torch.set_default_device(gpu)

    # Training loop
    model = Encoder()
    model.to(gpu)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,lambda epoch:0.99) # Don't actually use this anymore but I left it there for a print statemen

    tensor_list = torch.load('models/tensor_count')
    tensor_list = tensor_list.to(gpu)

    val_best = 100000
    best_val_acc = 0
    val_not_improve_count = 0 # Was using this to implement early stopping but got rid of it.  Left it in as extra information.
    loss_list = []
    for epoch in range(1000):
        model.train()
        train_loss = 0
        for x in range(int(len(data)/20)):
            sample = data[x*20:(x+1)*20].to(gpu)
            result = model.forward(sample,tensor_list)
            loss = loss_fn(result,sample)
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            print(f'training {x} of {len(data)/20}           ',end='\r')
        loss_list.append(train_loss)
        if train_loss < val_best:
            val_best = train_loss
            val_not_improve_count = 0
            torch.save(model.state_dict(),f'{model.name()}')
        elif epoch > 20:
            val_not_improve_count += 1
        print(f'epoch {epoch} lr {scheduler.get_lr()} : count {val_not_improve_count} : train loss {int(train_loss*1000)} : best {int(val_best*1000)}                                                        ')
        if epoch >= 20:
                optimizer.param_groups[0]['lr'] = 0.0001 *(1020-epoch)/1000 # custom LR schedule

        gc.collect()

if __name__ == '__main__':
    vector_run(MLP_RNN_voter_model,50)