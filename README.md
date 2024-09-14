# c_abbott_portfolio
Thank you for visiting my portfolio.  Below is a description of each of the sample projects in this repository.

### GNN PHYSICAL DYNAMICS DISCOVERY ###

This project was part of a graduate-level deep learning class. Graph Neural Networks (GNNs) are increasingly effective at modeling real-world physical dynamics. Following research from Google's DeepMind group, I trained GNNs to simulate various system dynamics. I then experimented with different neural network designs to analyze the learned parameters and extract information about the dynamics.

### Potential Application ###: Remote scientific discovery. For instance, a rover on a distant planet could film particles as they are disturbed, and training a GNN on this data could reveal properties like magnetism.

Result: Promising findings despite limited training. While the GNNs were not ideal simulators due to restricted training cycles, they learned enough interactions that an analyzing network achieved over 90% accuracy in identifying interaction types.

Included in Folder: Examples of the trained GNNs on gas diffusion simulations, the neural network used for parameter analysis, and the final project write-up.

### RETRIEVAL AUGMENTED GENERATION RESEARCH CHAT ###

This project was a proof of concept for a potential professional engagement. The goal was to create a chat system that could reference scientific papers during conversations. All research papers in the Physics section of ArXiv were downloaded, chunked, encoded, and stored in a Milvus vector database. Users could then interact with a Large Language Model (LLM), specifically LLaMA 3, and the system would retrieve relevant segments of research papers to provide context for the model’s responses.

Technology: The project was initially set up using the Nvidia DGX cloud, with plans for local deployment in production.

Result: Successfully demonstrated a retrieval-augmented generation process that could assist researchers in navigating and summarizing vast scientific literature.

Potential Next Steps: Integration with a more diverse set of databases and refinement for real-world use cases in scientific research.

### TASTYTRADE SPX OPEN BUTTERFLY ###

This personal project utilized TastyTrade APIs to automate stock trading. The program operated 24/7, initiating a butterfly options position on the SPX at the market open each day. Alternative versions used neural network and random forest models, fed with the previous 30 days of SPX movement, to trigger directionally biased trades.

Technology: TastyTrade API, neural networks, and random forest models for trade decision-making.

Result: The program performed reliably, running continuously as designed. While the trading strategies themselves were not highly successful, this project was an invaluable learning experience. It taught me how to build robust, always-on software and introduced me to algorithmic trading.

Lessons Learned: The importance of handling real-time data, implementing fail-safes, and refining algorithms for continuous operation.

### AZURE DUBBING DEMO ###

This was a rapid proof-of-concept project developed for a customer. The aim was to demonstrate how 911 calls from non-English speakers could be automatically transcribed, translated into English for the operator, and then have the operator’s response translated back into the caller's language with voice synthesis. I created this demo in a single afternoon using Azure’s cognitive services.

Potential Application: Enhancing emergency services by providing instant translation and transcription for non-English-speaking callers.

Result: A basic but functional "art of the possible" demo. A more refined proof of concept was later developed using Meta's Seamless model, though Meta declined a license exception for further use.  This project is currently in development with this customer using the Nvidia AI Enterprise ecosystem and the Nvidia Riva models for transcription, translation, and voice synthesis.

Lessons Learned: Demonstrated the potential of AI in real-time language translation and identified limitations related to licensing and deployment of advanced models.
