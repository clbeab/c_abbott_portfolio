import mujoco
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime as dt

def get_distances(position_list,edge_list,boundary_pos=np.array((0,0,0,0,0,0))):
  '''Positions: Just the most recent time step of positions \n
  Edge List: {sender,reciever,rigid value}\n
  Returns dict of edge lists with {r,s} as key\n
  Boundary defaults to just floor at z=0.  Other boundaries need to be in format (-z,+z,+x,-x,+y,-y). 0's will be treated as no boundary.'''
  # edge_array {sender,reciever,|dist|,ref dist,x,y,z}
  edge_dict = dict()
  # rel_pos_array gives relative position to each neighbor as [vertex,neighbor,dim]
  dist_to_b = np.zeros((position_list.shape[0],6))
  for s in range(edge_list.shape[0]):
    for r in range(edge_list.shape[0]):
      if edge_list[s,r] != 0:
        s_pos = position_list[s,:]
        r_pos = position_list[r,:]
        rel_pos = s_pos-r_pos
        abs_dist = np.linalg.norm(rel_pos)
        if edge_list[s,r] != 1:
          rig_dist = edge_list[s,r]
        else:
          rig_dist = abs_dist
        edge_dict[f'{s},{r}'] = np.array((abs_dist,rig_dist,rel_pos[0],rel_pos[1],rel_pos[2]))
  for node in range(position_list.shape[0]):
    x = position_list[node,0]
    y = position_list[node,1]
    z = position_list[node,2]
    dist_to_b[node,0] = np.max(1-z,0)
    if boundary_pos[1] != 0:
      dist_to_b[node,1] = np.max(1+z-boundary_pos[1],0)
    if boundary_pos[2] != 0:
      dist_to_b[node,2] = np.max(1+x-boundary_pos[2],0)
    if boundary_pos[3] != 0:
      dist_to_b[node,3] = np.max(1-x+boundary_pos[3],0)
    if boundary_pos[4] != 0:
      dist_to_b[node,4] = np.max(1+y-boundary_pos[4],0)
    if boundary_pos[5] != 0:
      dist_to_b[node,5] = np.max(1-y+boundary_pos[5],0)
  return edge_dict, dist_to_b

def generate_box_model_data(velocities,friction, edge_length = 0.4):
  '''Velocities: {x,y,z,rx,ry,rz} sets initial velocities for model cube\n
  Friction: float values between 0 and 1'''

  xml = f"""
  <mujoco>
    <asset>
      <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
      rgb2=".2 .3 .4" width="300" height="300" mark="none"/>
      <material name="grid" texture="grid" texrepeat="1 1"
      texuniform="true" reflectance=".2"/>
    </asset>
    <worldbody>
      <light name="top" pos="0 0 10"/>
      <geom name="floor" type="plane" pos="0 0 -.5" size="2 2 .1" material="grid"/>
      <body name="cube" pos="0 0 0">
        <joint name="free" type="free"/>
        <geom name="vert1" type="sphere" size=".01 .01 .01" rgba="1 0 0 1" pos="0 0 0" friction="{friction}"/>
        <geom name="vert2" type="sphere" size=".01 .01 .01" rgba="1 0 0 1" pos="{edge_length} 0 0" friction="{friction}"/>
        <geom name="vert3" type="sphere" size=".01 .01 .01" rgba="1 0 0 1" pos="0 {edge_length} 0" friction="{friction}"/>
        <geom name="vert4" type="sphere" size=".01 .01 .01" rgba="1 0 0 1" pos="{edge_length} {edge_length} 0" friction="{friction}"/>
        <geom name="vert5" type="sphere" size=".01 .01 .01" rgba="1 0 0 1" pos="0 0 {edge_length}" friction="{friction}"/>
        <geom name="vert6" type="sphere" size=".01 .01 .01" rgba="1 0 0 1" pos="{edge_length} 0 {edge_length}" friction="{friction}"/>
        <geom name="vert7" type="sphere" size=".01 .01 .01" rgba="1 0 0 1" pos="0 {edge_length} {edge_length}" friction="{friction}"/>
        <geom name="vert8" type="sphere" size=".01 .01 .01" rgba="1 0 0 1" pos="{edge_length} {edge_length} {edge_length}" friction="{friction}"/>
        <geom name="box" type="box" size=".2 .2 .2" rgba="1 0 0 1" pos=".2 .2 .2" friction="{friction}"/>
      </body>
    </worldbody>

    <keyframe>
      <key name="toss" qpos="0 0 0 0 0 0 0" qvel="{velocities[0]} {velocities[1]} {velocities[2]} {velocities[3]} {velocities[4]} {velocities[5]}"/>
    </keyframe>
  </mujoco>
  """

  # List of all edges dim1 = vertex, dim2 = neighbor
  edges = np.array(((2,3,5),(1,4,6),(4,1,7),(3,2,8),(6,7,1),(5,8,2),(8,5,3),(7,6,4)))
  model = mujoco.MjModel.from_xml_string(xml)
  data = mujoco.MjData(model)

  # Set the number of data points as "frames" and duration as frames/fps
  n_frames = 100
  fps = 60.0

  # Initiate Model
  mujoco.mj_forward(model, data)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  positions = np.zeros((n_frames,8,3))

  # Generate model mata and transcribe to positions
  for i in range(n_frames):
    while data.time < i/fps:
      mujoco.mj_step(model,data)
    for v in range(8):
      for x in range(3):
        # Populates the position array (pos) with the x,y,z coordinates of each vertices
        positions[i,v,x] = data.geom(f'vert{v+1}').xpos[x]
  positions = positions[1:,:,:]
  positions[:,:,2] = positions[:,:,2] + .5
  edge_list = np.zeros((8,8))
  for s in range(edges.shape[0]):
    for r in range(3):
      edge_list[s,edges[s,r]-1] = 1
  return positions, edge_list, edge_length

def get_velocities(position_list):
  '''Requires previous 6 positions {-6:,:,:}\n
  Returns array(number of vertices,5,3) = {node, time, xyz vel} where time[0] is the current velocity'''
  if position_list.shape[0] == 6:
    pass
  else:
    raise ValueError('get_velocities requires the last 6 position lists')
  # Calulate the current and 4 previous velocities in each dimension for each vertex
  velocities = np.zeros((position_list.shape[1],5,3))
  for vert in range(position_list.shape[1]):
    for dim in range(3):
      for time in range(5):
        velocities[vert,time,dim] = (position_list[5-time,vert,dim] - position_list[5-time-1,vert,dim])/0.002
  return velocities
  
def create_graph(positions,velocities,node_attribute_list,distances,relative_positions,globals=None):
  num_nodes = positions.shape[0]
  nodes = np.zeros((num_nodes,40))
  num_edges = relative_positions.shape[1]
  edges = np.zeros((num_nodes,num_edges*3+(distances.shape[1]-5)))
  for node in range(num_nodes):
    nodes[node,0:3] = positions[node,:]
    nodes[node,3:8] = velocities[node,:,0]
    nodes[node,8:13] = velocities[node,:,1]
    nodes[node,13:18] = velocities[node,:,2]
    nodes[node,18:34] = node_attribute_list[node,:]
    nodes[node,34:39] = distances[node,-5:]
    edges[node,:num_edges] = relative_positions[node,:,0]
    edges[node,num_edges:2*num_edges] = relative_positions[node,:,1]
    edges[node,2*num_edges:3*num_edges] = relative_positions[node,:,2]
    edges[node,-num_edges:] = distances[node,:num_edges]
  return nodes, edges, globals
   
def get_node_list(positions,velocities,dist_to_boundaries):
  node_list = np.zeros((positions.shape[0],24))
  for node in range(positions.shape[0]):
    node_list[node,:3] = positions[node,:]
    for t in range(5):
      node_list[node,3+3*t:6+3*t] = velocities[node,t,:]
    node_list[node,18:24] = dist_to_boundaries[node,:]
  return node_list

def update_nodes(positions, velocities, new_accelerations):
  new_nodes = np.zeros_like(positions)
  new_accelerations = new_accelerations.cpu()
  new_accelerations = new_accelerations.detach().numpy()
  for node in range(positions.shape[0]):
    for dim in range(3):
      old_pos = positions[node,dim]
      old_v = velocities[node,0,dim]
      acc = new_accelerations[node,dim]
      new_nodes[node,dim] = old_pos + 0.002*(old_v + 0.002*acc)
  return new_nodes

def get_acceleration_targets(positions, t):
  acc = positions[t+1,:,:] - 2*positions[t,:,:] + positions[t-1,:,:]
  return acc

def flatten_to_loss_input(input):

  return input  

class GNN(nn.Module):
  def __init__(self,attributes,global_u) -> None:
    super().__init__()
    self.atrribs = nn.Parameter(attributes, requires_grad=True)
    self.u = nn.Parameter(global_u, requires_grad=True)
    self.ev = nn.Linear(40,128)
    self.ee = nn.Linear(5,128)
    self.pe1 = nn.Linear(400,128)
    self.pea1 = nn.ReLU()
    self.pe2 = nn.Linear(128,128)
    self.pea2 = nn.ReLU()
    self.pe3 = nn.Linear(128,128)
    self.pea3 = nn.ReLU()
    self.pe4 = nn.Linear(128,128)
    self.pv1 = nn.Linear(272,128)
    self.pva1 = nn.ReLU()
    self.pv2 = nn.Linear(128,128)
    self.pva2 = nn.ReLU()
    self.pv3 = nn.Linear(128,128)
    self.pva3 = nn.ReLU()
    self.pv4 = nn.Linear(128,128)
    self.pu1 = nn.Linear(272,128)
    self.pua1 = nn.ReLU()
    self.pu2 = nn.Linear(128,128)
    self.pua2 = nn.ReLU()
    self.pu3 = nn.Linear(128,128)
    self.pua3 = nn.ReLU()
    self.pu4 = nn.Linear(128,16)
    self.d1 = nn.Linear(128,128)
    self.da1 = nn.ReLU()
    self.d2 = nn.Linear(128,128)
    self.da2 = nn.ReLU()
    self.d3 = nn.Linear(128,128)
    self.da3 = nn.ReLU()
    self.d4 = nn.Linear(128,3)

  def forward(self,edge_list,ev_in,ee_in,pu_in):
    nodes = edge_list.shape[0]
    ev_in = torch.tensor(ev_in).float()
    ev_in = torch.concat((ev_in,self.atrribs),dim=1)
    ee_in = torch.tensor(ee_in).float()
    u = self.u
    # ENCODE
    v = self.ev(ev_in)
    e = self.ee(ee_in)
    ep = []
    # PROCESS
    for message_pass in range(3):
      ep = []
      # phi e MLP
      edge = 0
      for x in range(nodes):
        for y in range(nodes):
          if edge_list[x,y] != 0:
            input = torch.concat((e[edge,:],v[x,:],v[y,:],u))
            ep.append(input)
            edge += 1
      ep = torch.stack(ep)
      ep = self.pe1(ep)
      ep = self.pea1(ep)
      ep = self.pe2(ep)
      ep = self.pea2(ep)
      ep = self.pe3(ep)
      ep = self.pea3(ep)
      ep = self.pe4(ep)
      edge = 0
      ep_comb = []
      node_counter = 0
      for x in range(nodes):
        temp_tensor = []
        for y in range(nodes):
          if edge_list[x,y] != 0:
            if node_counter == 0:
              temp_tensor.append(ep[edge])
              node_counter = 1
            else:
              temp_tensor[0] = torch.add(temp_tensor[0],ep[edge])
            edge += 1
        if node_counter == 0:
          ep_comb.append(torch.zeros_like(ep[0]))
        else:
          ep_comb.append(temp_tensor[0])
        node_counter = 0
      ep_comb = torch.stack(ep_comb)
      vp = []
      # phi v MLP
      # print(ep_comb[0])
      for node in range(nodes):
        input = torch.concat((ep_comb[node],v[node],u))
        vp.append(input)
      vp = torch.stack(vp)
      vp = self.pv1(vp)
      vp = self.pva1(vp)
      vp = self.pv2(vp)
      vp = self.pva2(vp)
      vp = self.pv3(vp)
      vp = self.pva3(vp)
      vp = self.pv4(vp)
      e_bar = torch.sum(ep_comb,dim=0)
      v_bar = torch.sum(vp,dim=0)
      # phi u MLP
      input = torch.concat((e_bar,v_bar,u))
      input = self.pu1(input)
      input = self.pua1(input)
      input = self.pu2(input)
      input = self.pua2(input)
      input = self.pu3(input)
      input = self.pua3(input)
      u = self.pu4(input)
      e = ep
      v = vp
    #DECODE
    y = self.d1(vp)
    y = self.da1(y)
    y = self.d2(y)
    y = self.da2(y)
    y = self.d3(y)
    y = self.da3(y)
    y = self.d4(y)
    return y

def box_model(reference=False):
  if reference == True:
    torch.manual_seed(132)
  random_nums = (np.random.rand((6))-.5)*5
  edge_length = 0.4
  positions, edge_list, edge_length = generate_box_model_data(random_nums,.5,edge_length)
  edge_list = edge_list*edge_length
  return positions, edge_list

def get_edge_list(positions,dist):
  nodes = positions.shape[0]
  edge_list = np.zeros((nodes,nodes))
  for x in range(nodes):
    for y in range(nodes):
      if np.linalg.norm((positions[x,:]-positions[y,:])) < dist and x!=y:
        edge_list[x,y] = 1
  return(edge_list)

torch.set_default_device(torch.device('cuda'))
positions = np.load('Python Code\Data\DiffusionSims/0.npy') - 55
positions = positions[::5]/5
### DATA GENERATION ABOVE THIS POINT ###

### MODEL RUN BELOW ###
nodes = positions.shape[1]
attributes = torch.rand((nodes,16))
global_u = torch.rand((16)).float()
model = GNN(attributes,global_u)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
loss_fn =nn.MSELoss()
losses = []
best_loss = 1
model_start = dt.now()
for epoch in range(1000):
  start_time = dt.now()
  optimizer.zero_grad()
  preds = []
  targets = []
  actual_pos = []
  pred_pos = []
  for t in range(5,30):
    # True for dynamic systems, False for rigid
    if 1 == 1:
      edge_list = get_edge_list(positions[t,:,:],2)
    targets.append(torch.tensor(get_acceleration_targets(positions,t)).float())
    edge_dict, dist_to_boundaries = get_distances(positions[t,:,:],edge_list,np.array((-10,10,10,-10,10,-10)))
    edge_input = []
    for key in edge_dict:
      edge_input.append(edge_dict[key])
    edge_input = np.array(edge_input)
    print(edge_input)
    velocities = get_velocities(positions[t-5:t+1,:,:])
    node_list = get_node_list(positions[t,:,:],velocities,dist_to_boundaries)
    accels = model.forward(edge_list,node_list,edge_input,pu_in=global_u)
    preds.append(accels)
    predicted_pos = update_nodes(positions[t,:,:],velocities,accels)
    pred_pos.append(predicted_pos)
    actual_pos.append(positions[t+1,:,:])
    print(f't[{t+1}/30]',end='\r')
  targets = torch.stack(targets)
  preds = torch.stack(preds)
  loss = loss_fn(preds,targets)
  loss.backward()
  optimizer.step()
  pred_pos = np.array(pred_pos)
  actual_pos = np.array(actual_pos)
  loss = loss.cpu()
  losses.append(loss.detach().numpy())
  ref = optimizer.param_groups[0]['lr']
  print(f'Epoch {epoch+1}:lr {ref} finished in {dt.now() - start_time} with loss: {loss}')
  if loss < best_loss:
    best_loss = loss
  if loss < 1e-7:
    optimizer.param_groups[0]['lr'] = 0.00001
  elif loss < 3e-6:
    optimizer.param_groups[0]['lr'] = 0.0001

print('                                                                                      ',end='\r')
print(f'Model finished in ({dt.now()-model_start}) with loss: {best_loss}')

### PLOTTING ###
if 1 == 1:
    fig, axs = plt.subplots(3,2)
    fig.suptitle('Node 0: Prediction vs Actual')
    preds = preds.cpu()
    targets = targets.cpu()
    axs[0,0].plot(preds[:,0,0].detach().numpy())
    axs[0,0].plot(targets[:,0,0].detach().numpy())
    axs[0,0].legend(('predicted','actual'))
    axs[0,0].set_title('Acceleration')
    axs[0,0].set_ylabel('X')
    axs[1,0].plot(preds[:,0,1].detach().numpy())
    axs[1,0].plot(targets[:,0,1].detach().numpy())
    axs[1,0].legend(('predicted','actual'))
    axs[1,0].set_ylabel('Y')
    axs[2,0].plot(preds[:,0,2].detach().numpy())
    axs[2,0].plot(targets[:,0,2].detach().numpy())
    axs[2,0].legend(('predicted','actual'))
    axs[2,0].set_ylabel('Z')
    axs[0,1].plot(pred_pos[:,0,0])
    axs[0,1].plot(actual_pos[:,0,0])
    axs[0,1].legend(('predicted','actual'))
    axs[0,1].set_title('Position')
    axs[1,1].plot(pred_pos[:,0,1])
    axs[1,1].plot(actual_pos[:,0,1])
    axs[1,1].legend(('predicted','actual'))
    axs[2,1].plot(pred_pos[:,0,2])
    axs[2,1].plot(actual_pos[:,0,2])
    axs[2,1].legend(('predicted','actual'))
    plt.show()
    plt.plot(np.array(losses))
    plt.show()





