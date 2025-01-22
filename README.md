# Balancing State Exploration and Skill Diversity in Unsupervised Skill Discovery

## Code
This article is currently under revision, and the code will be organized and open-sourced after acceptance.




## Videos of the discovered unsupervised skills


### ComSD (ours)


ComSD can discover **diverse behaviors at different activity levels** without extrinsic rewards, including both dynamic movements and static postures. **(balancing state exploration and skill diversity)**


We visualized some representative skills discovered by ComSD in **Walker** and **Cheetah**. Note that the skill discovery stage is totally task-agnostic, and all the discovered skills are pre-trained with only intrinsic rewards. Above the videos are the descriptions of the discovered unsupervised skills, not specific tasks. For example, in **Walker**, the discovered behavior in the sixth column of the second row is 'kneel down'.

#### Walker
| Flip forward | Flip backward | Flip failed  | Advance on knees | Kick back | Crawl forward |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Shake the left leg** | **Stand up** | **Flip to yoga** | **Lift the right leg** | **Lift the left leg** | **Kneel down** |


|![GIF 1](comsdgif/w12.gif) |![GIF 2](comsdgif/w11.gif) |![GIF 3](comsdgif/w10.gif) |![GIF 4](comsdgif/w9.gif) |![GIF 5](comsdgif/w8.gif) |![GIF 6](comsdgif/w7.gif) |
| :---: | :---: | :---: | :---: | :---: | :---: |
|![GIF 7](comsdgif/w6.gif) |![GIF 8](comsdgif/w5.gif) |![GIF 9](comsdgif/w4.gif) |![GIF 10](comsdgif/w3.gif) |![GIF 11](comsdgif/w2.gif) |![GIF 12](comsdgif/w1.gif) |



#### Cheetah


| Flip backward | Jump | Flip forward | Walk on hands | Step back | Walk forward |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Shake hand** | **Leg taps floor** |  **Leg taps floor(slow)** | **Hand taps floor** | **Posture 1** | **Posture 2** |




|![GIF 1](comsdgif/c12.gif) |![GIF 2](comsdgif/c11.gif) |![GIF 3](comsdgif/c10.gif) |![GIF 4](comsdgif/c9.gif) |![GIF 5](comsdgif/c8.gif) |![GIF 6](comsdgif/c7.gif) |
| :---: | :---: | :---: | :---: | :---: | :---: |
|![GIF 7](comsdgif/c6.gif) |![GIF 8](comsdgif/c5.gif) |![GIF 9](comsdgif/c4.gif) |![GIF 10](comsdgif/c3.gif) |![GIF 11](comsdgif/c2.gif) |![GIF 12](comsdgif/c1.gif) |


___
Recent advanced approaches cannot take both state exploration and skill diversity into account. They also can't generate behaviors at different activity levels.


### CIC


CIC is able to produce dynamic movements, but the generated skills are **indistinguishable and homogeneous**.  In addition, it can't generate behaviors at other activity levels (e.g., postures).  **(high state exploration, low skill diversity)**




#### Left three: Walker &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Right three: Cheetah

 | | | | | | |
| :---: | :---: | :---: | :---: | :---: | :---: |
|![GIF 1](cic/w1.gif)<br>|![GIF 2](cic/w2.gif)<br>|![GIF 3](cic/w3.gif)<br> |![GIF 4](cic/c1.gif)<br> |![GIF 5](cic/c2.gif)<br>|![GIF 6](cic/c3.gif)<br> |

Skills discovered by CIC are all trying to dynamically flip.



### APS


APS can generate non-homogeneous postures, but it suffers from **lazy state exploration**. It can't generate behaviors at other activity levels (e.g., dynamic flip). **(high skill diversity, low state exploration)**



#### Left three: Walker &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Right three: Cheetah
| | | | | | |
| :---: | :---: | :---: | :---: | :---: | :---: |
|![GIF 1](aps/w1.gif)<br> |![GIF 2](aps/w2.gif)<br>|![GIF 3](aps/w3.gif)<br>|![GIF 4](aps/c1.gif)<br> |![GIF 5](aps/c2.gif)<br>|![GIF 6](aps/c3.gif)<br>|

Skills discovered by APS are all different static postures.

___

The quantitative analysis also verify that our ComSD enables better balance between diversity and exploration than advanced baselines.


![Example Image](quantitative.png)





