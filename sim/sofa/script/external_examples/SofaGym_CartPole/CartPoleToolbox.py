"""
The next step is to implement the methods and functions necessary for the RL algorithms to work. 
Three essential parts need to be implemented for this, defining and applying the actions, 
defining and calculating the reward, and getting the new observation from the simulation and updating it. 
One additional component that is relevant to some environments is a GoalSetter to define and update a target goal if necessary.

First, we need to create a new file CartPoleToolbox. 
For each of the parts that need to be implemented, some components are required to be defined to make the SOFA scene compatible with SofaGym.

Actions:
    startCmd function
Reward:
    getReward function
    RewardShaper class
Observation:
    getState function
    getPos function
    setPos function
Goal:
    GoalSetter class
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

import math

import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
from splib3.animation.animate import Animation

SofaRuntime.importPlugin("Sofa.Component")

class ApplyAction(Sofa.Core.Controller):
    """
    We can define an ApplyAction class that inherits from Sofa.Core.Controller, 
    which allows us to add this component to the scene and make it update the value of the constantForceField component during the simulation steps. 
    We initialize an incr variable to define the value of the force change between two consecutive steps.

    The compute_action method sets the actual value of the action to take based on the value returned by the RL algorithm. 
    he _move method is used to update the force field applied on the cart by changing the x value of the constantForceField object of the cart 
    to control the motion of the cart left and right. 
    Finally, the apply_action method will be used to apply the action by the agent.
    """
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root = kwargs["root"]
        self.cart = self.root.Modeling.Cart

        self.incr = 1000

    def _move(self, incr):
        cartForceField = self.cart.CartForce
        force = cartForceField.force.value.tolist()
        force[0] = incr
        cartForceField.force.value = np.array(force)

    def compute_action(self, actions):
        if actions == 0:
            incr = self.incr
        else:
            incr = -self.incr

        return incr

    def apply_action(self, incr):
        self._move(incr)

"""
Next, the required part is to define the startCmd function, which is used by SofaGym as a link between Gym and SOFA to execute the actions 
as SOFA commands in the simulation. 
The ApplyAction class we defined is used here.

Two helper functions can be defined first. 
The action_to_command function returns the value of the action using compute_action. 
The startCmd_CartPole function is used to call the SOFA AnimationManager to execute the necessary command in the simulation based on the chosen action. 
By defining these two functions, it is simple to define the startCmd function to get the needed command and apply it to the simulation.
"""
def action_to_command(actions, root, nb_step):
    """Link between Gym action (int) and SOFA command (displacement of cables).

    Parameters:
    ----------
        action: int
            The number of the action (Gym).
        root:
            The root of the scene.

    Returns:
    -------
        The command.
    """

    incr = root.ApplyAction.compute_action(actions)
    return incr


def startCmd(root, actions, duration):
    """Initialize the command from root and action.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.
        action: int
            The action.
        duration: float
            Duration of the animation.

    Returns:
    ------
        None.

    """
    incr = action_to_command(actions, root, duration/root.dt.value + 1)
    startCmd_CartPole(root, incr, duration)


def startCmd_CartPole(rootNode, incr, duration):
    """Initialize the command.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.
        incr:
            The elements of the command.
        duration: float
            Duration of the animation.

    Returns:
    -------
        None.
    """

    # Definition of the elements of the animation
    def executeAnimation(rootNode, incr, factor):
        rootNode.ApplyAction.apply_action(incr)

    # Add animation in the scene
    rootNode.AnimationManager.addAnimation(
        Animation(
            onUpdate=executeAnimation,
            params={"rootNode": rootNode,
                    "incr": incr},
            duration=duration, mode="once"))
    

"""
For the reward, we define a RewardShaper class to inherit from Sofa.Core.Controller to update the reward value at each simulation step. 
In the initialization, we can define some parameters based on the scene configs, such as the pole length (pole_length) and the maximum angle (max_angle) 
the pole is allowed to tilt.

Depending on the scene, some helper methods could be defined to get or calculate different values. 
For the CartPole scene, we need to get the pole's x and y positions and the cart's x position. We also need to calculate the pole's angle and angular velocity. 
Finally, the getReward method uses these helper methods to calculate and return the reward, as well as, the necessary states to determine whether 
or not a termination condition happened. 
Since the reward is 1 for each step the pole stays within the max angle limit, we simply return 1 as the reward, 
in addition to the current pole angle and the max allowed angle.

An update method is also required to initialize the reward to specific values at each episode reset. In this case, it is not needed.
"""
class RewardShaper(Sofa.Core.Controller):
    """Compute the reward.

    Methods:
    -------
        __init__: Initialization of all arguments.
        getReward: Compute the reward.
        update: Initialize the value of cost.

    Arguments:
    ---------
        rootNode: <Sofa.Core>
            The scene.

    """
    def __init__(self, *args, **kwargs):
        """Initialization of all arguments.

        Parameters:
        ----------
            kwargs: Dictionary
                Initialization of the arguments.

        Returns:
        -------
            None.

        """
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.rootNode = None
        if kwargs["rootNode"]:
            self.rootNode = kwargs["rootNode"]
        if kwargs["max_angle"]:
            self.max_angle = kwargs["max_angle"]
        if kwargs["pole_length"]:
            self.pole_length = kwargs["pole_length"]
        
        else:
            print(">> ERROR: give a max angle for the normalization of the reward.")
            exit(1)

        self.cart = self.rootNode.Modeling.Cart
        self.pole = self.rootNode.Modeling.Pole

    def getReward(self):
        """Compute the reward.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The reward and the cost.

        """
        #dist = abs(pole_x_pos - cart_pos)


        cart_pos = self._getCartPos()
        pole_theta, pole_theta_dot = self.calculatePoleTheta(cart_pos)
        
        return 1, pole_theta, self.max_angle

    def update(self):
        """Update function.

        This function is used as an initialization function.

        Parameters:
        ----------
            None.

        Arguments:
        ---------
            None.

        """
        pass

    def _getPolePos(self):
        pos = self.pole.MechanicalObject.position.value.tolist()[0]
        return pos[0], pos[1]

    def _getCartPos(self):
        pos = self.cart.MechanicalObject.position.value.tolist()[0][0]
        return pos
    
    def calculatePoleTheta(self, cart_pos):
        x_pos, y_pos = self._getPolePos()
        sin_theta = (y_pos/self.pole_length)
        theta = abs((90*math.pi/180) - math.asin(sin_theta))
        
        if x_pos < cart_pos:
            theta = -theta

        theta_dot = self.pole.MechanicalObject.velocity.value.tolist()[0][5]
        
        return theta, theta_dot
    

"""
The getReward function must be defined to be used by SofaGym to calculate the reward value and done state and return them by Gym at each step. 
We simply use the RewardShaper class we just defined to get the reward and check if the episode is done based on the termination condition. 
For the CartPole, the episode ends when the pole angle theta exceeds the allowed limit (24 degrees as defined in the environment's config) in the left or right 
direction.
"""

def getReward(rootNode):
    reward, theta, max_angle = rootNode.Reward.getReward()
    done = (theta > max_angle) or (theta < -max_angle)

    return done, reward


"""
After applying the action to the simulation scene and calculating the returned reward, the new state of the environment must also be returned. 
To do this, it is required to define a getState function to get and calculate the new state and return it. 
As discussed in Step 1, the CartPole environment's state at each step consists of 4 values of the cart's x position and velocity, 
and the pole's angle and angular velocity. 
We can simply get the cart's state from the SOFA scene cart object, and we can use the method we previously defined in the Reward class to get the pole's state.
"""
def getState(rootNode):
    """Compute the state of the environment/agent.

    Parameters:
    ----------
        rootNode: <Sofa.Core>
            The scene.

    Returns:
    -------
        State: list of float
            The state of the environment/agent.
    """
    cart = rootNode.Modeling.Cart
    pole = rootNode.Modeling.Pole

    cart_pos = cart.MechanicalObject.position.value.tolist()[0][0]
    cart_vel = cart.MechanicalObject.velocity.value.tolist()[0][0]

    pole_theta, pole_theta_dot = rootNode.Reward.calculatePoleTheta(cart_pos)

    state = [cart_pos, cart_vel, pole_theta, pole_theta_dot]

    return state

"""
The second required part is to define two functions: getPos and setPos to retrieve the cart and pole positions at the beginning of each step and 
update them at the end of the step with the new state. These two functions are mainly needed for the rendering part.
"""
def getPos(root):
    cart_pos = root.Modeling.Cart.MechanicalObject.position.value[:].tolist()
    pole_pos = root.Modeling.Pole.MechanicalObject.position.value[:].tolist()
    return [cart_pos, pole_pos]


def setPos(root, pos):
    cart_pos, pole_pos = pos
    root.Modeling.Cart.MechanicalObject.position.value = np.array(cart_pos)
    root.Modeling.Pole.MechanicalObject.position.value = np.array(pole_pos)

"""
This step is only feasible for environments or scenes where a goal is defined, such as a target position that a robot needs to reach. 
The GoalSetter class can then be used to initialize the goal in the scene and update it at each episode or randomly change it for example.
Since the CartPole scene doesn't require this, no modifications to the goal class are needed.
"""

class GoalSetter(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

    def update(self):
        pass

    def set_mo_pos(self, goal):
        pass