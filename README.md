# PixelCopter Bot

Reinforcement learning agent that learns to play PixelCopter using Q-learning with neural network function approximations. 

![gif](https://github.com/eddieshengyuwang/Heli-Reinforcement-Learning/blob/master/heli.gif)

Developed this project to learn the basics of RL, Q-learning with NNs, and using Keras (Theano backend). As for running simulations, I used the [Python Learning Environment](https://pygame-learning-environment.readthedocs.io/en/latest/), which has a number of pre-built games in Python (like Flappy Bird, Pong, PixelCopter, etc). This way, I didn't have to build the game but instead focus more on designing the agent, which is awesome!

## Resources

I read a variety of articles and Medium blog posts to get familiar with RL. There are also a lot of Github implementations of other bots, like Flappy Bird, which I also used as a reference (but I couldn't find any for PixelCopter...)

Here are some links:

- [Q-learning with Neural Networks](http://outlace.com/rlpart3.html) - this is the one I used the most, it's amazing. I recommend understanding parts 1 and 2 also.
- [An introduction to Deep Q-Learning: let’s play Doom](https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8)
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602v1.pdf) - DeepMind Paper
- [David Silver's RL Course](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-) - I watched a few of these
- [Python Learning Environment](https://pygame-learning-environment.readthedocs.io/en/latest/) - used to build the game

## Notes on the model

- The bot uses a simple 2-hidden layer NN built with Keras with Theano backend (5 input nodes, 20 nodes in first hidden layer, 10 nodes in second hidden layer, and 2 output nodes).

- 5 input units are given through PLE's ```getGameState``` method, which returns features like y position, velocity, distance to ceil, etc.

- Initially, PixelCopter has random map generation per new episode and green blocks that the white pixel has to dodge. I removed both of these because training was taking too long to converge with the added complexity.

- Training took around 8 hours for ~10,000 episodes on an old Dell Precision. It might have taken less but I just left it overnight.

## How to run

Make sure to have Keras installed and running with Theano backend. Also have pygame and PLE installed. Then run the notebook on Jupyter. The game will run by using the most recently updated weights for the NN. To train the model from scratch again, follow uncomment/comment directions in the notebook.

## Next steps
- Expand PixelCopter map to run longer
- Replicate RL agent but use a CNN to use pixel-by-pixel game frames as input instead of relying on PLE's methods to provide inputs

