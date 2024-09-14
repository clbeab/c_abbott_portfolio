# c_abbott_portfolio
Thank you for visiting my portfolio.  Below is a description of each of the sample projects in this repository.

### GNN PHYSICAL DYNAMICS DISCOVERY ###

This project was done for a graduate level Deep Learning class.  Graph Neural Network simulations have gotten very good at modeling real world physical system dynamics.  In this work, I trained Graph Neural Networks to simulate various system dynamics following the work done by the Google DeepMind group.  I then used various different neural network designs to analayze the learned parameters to see if we could extract information about what type of dynamics were learned.

Potential appliactions for this project are remote scientific discovery.  As an example, if a rover on a distant planet found a material that we wanted to determine the properties of, such as magnetism, the rover would simply need to film the particles as they were disturbed.  Training a GNN on the filmed partcile interactions would then allow us to determine the properties of interest.

Result:  Actually quite promising results given the lack of time and resources.  The GNNS were effectively useless as simulators but enough of the interactions were learned in the parameters that the analyzing network was over 90% accurate in distinguishing the type of interaction.

Inf the folder is an example of one of the Graph Neural Networks trained on a gas diffusion simulation, the neural network that was used to analyze the learned parameters of the GNN, and the final project write up.

### RETRIEVAL AUGMENTED GENERATION RESEARCH CHAT ###

This project was done as a proof of concept for a potential professional engagement.  All the research papers in the Physics section of ArXiv would be downloaded, chunked, encoded, and loaded into a Milvus vector database.  Users could then chat with a LLM (Llama 3 in this instance) and the program would retrieve segments of research papers related to the question to be fed to the model during generation.  This was initially setup using the Nvidia DGX cloud with the intention of being deployed locally in production.

### TASTYTRADE SPX OPEN BUTTERFLY ###

This was a personal project that leveraged TastyTrade APIs to perform automated stock trading.  The program ran 24/7 and would initiate a butterfly options position on the SPX at the open of every day.  Other versions of this program used a neural network and random forest models fed with the previous 30 days of SPX movement to trigger a directionally biased trade.  The program performed wonderfully, the strategies not as much.  However, it was a very fun and useful project earlier in my programming development that taught me a lot about writing programs that function on a continuous basis instead of one shot and done.

### AZURE DUBBING DEMO ###

This was a very quick proof of concept I put together for a customer to show them how 911 calls from non-english speakers could be automatically transcribed and translated into english for the operator and the operators response translated and voice synthesized into the callers language.  It was a very rough "art of the possible" demo that I put together in a single afternoon after work.  A later version of this project was done using Meta's Seemless model in a single efficient step but Meta declined to make a license exception for that use case.
