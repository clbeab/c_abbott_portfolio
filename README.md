# c_abbott_portfolio
Thank you for visiting my portfolio.  Below is a description of each of the sample projects in this repository.

### GNN PHYSCIAL DYNMAICS DISCOVERY ###

This project was done for a graduate level Deep Learning class.  Graph Neural Network simulations have gotten very good at modeling real world physical system dynamics.  In this work, I trained Graph Neural Networks to simulate various system dynamics following the work done by the Google DeepMind group.  I then used various different neural network designs to analayze the learned parameters to see if we could extract information about what type of dynamics were learned.

Potential appliactions for this project are remote scientific discovery.  As an example, if a rover on a distant planet found a material that we wanted to determine the properties of, such as magnetism, the rover would simply need to film the particles as they were disturbed.  Training a GNN on the filmed partcile interactions would then allow us to determine the properties of interest.

Result:  Actually quite promising results given the lack of time and resources.  The GNNS were effectively useless as simulators but enough of the interactions were learned in the parameters that the analyzing network was over 90% accurate in distinguishing the type of interaction.
