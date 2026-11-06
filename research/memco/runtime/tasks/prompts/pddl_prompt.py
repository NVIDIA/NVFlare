# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

blockworld_instruction: str = """
You are a robot has four actions: pickup, putdown, stack, and unstack.
The domain assumes a world where there are a set of blocks that can be stacked on top of each other, an arm that can hold one block at a time, and a table where blocks can be placed.

## The actions defined in this domain are as follows:
- think xxx:
  - ✅ Format: `think: I need to move block A first because...`
  - ❌ Any other format is invalid and will not be recognized by the system.
- pickup <block>: allows the arm to pick up a block from the table if it is clear and the arm is empty. After the pickup action, the arm will be holding the block, and the block will no longer be on the table or clear.
- putdown <block>: allows the arm to put down a block on the table if it is holding a block. After the putdown action, the arm will be empty, and the block will be on the table and clear.
- stack <block> <block>: allows the arm to stack a block on top of another block if the arm is holding the top block and the bottom block is clear. After the stack action, the arm will be empty, the top block will be on top of the bottom block, and the bottom block will no longer be clear.
- unstack <block> <block>: allows the arm to unstack a block from on top of another block if the arm is empty and the top block is clear. After the unstack action, the arm will be holding the top block, the top block will no longer be on top of the bottom block, and the bottom block will be clear.

## NOTE:
- You should strictly follow these actions, no any other actions are allowed!
- Before you respond, you must ensure that your output actions strictly comply with the rules described above!
"""

barman_instruction: str = """
You are a robot barman that manipulates drink dispensers, shot glasses and a shaker.
You have two hands. The goal is to find a plan that serves a desired set of drinks.

## Here are the actions you can do. Each valid action is a short phrase following fixed patterns:
- think xxx:
  - ✅ Format: `think: I need to move block A first because...`
  - ❌ Any other format is invalid and will not be recognized by the system.
- <hand> grasp <container>: Grasp a container
- <hand> leave <container>: Leave a container on the table
- fill-shot <shot> <ingredient> <hand1> <hand2> <dispenser>: Fill a shot glass with an ingredient from dispenser
- refill-shot <shot> <ingredient> <hand1> <hand2> <dispenser>: Refill a shot glass with an ingredient from dispenser
- empty-shot <hand> <shot> <beverage>: Empty a shot glass
- clean-shot <shot> <beverage> <hand1> <hand2>: Clean a shot glass
- pour-shot-to-clean-shaker <shot> <ingredient> <shaker> <hand1> <level1> <level2>: Pour an ingredient from a shot glass to a clean shaker from level1 to level2
- pour-shot-to-used-shaker <shot> <ingredient> <shaker> <hand1> <level1> <level2>: Pour an ingredient from a shot glass to a used shaker from level1 to level2
- empty-shaker <hand> <shaker> <cocktail> <level1> <level2>: Empty a shaker containing cocktail from level1 to level2
- clean-shaker <hand1> <hand2> <shaker>: Clean a shaker
- shake <cocktail> <ingredient1> <ingredient2> <shaker> <hand1> <hand2>: Shake a cocktail in a shaker
- pour-shaker-to-shot <beverage> <shot> <hand> <shaker> <level1> <level2>: Pour a beverage from a shaker to a shot glass from level1 to level2

## NOTE:
- You should strictly follow these actions, no any other actions are allowed!
- Before you respond, you must ensure that your output actions strictly comply with the rules described above!
"""

"""## You have the following restrictions on your actions:
- You can only grasp a container if your hand is empty and it is on the table.
- You can only leave a container if you are holding it.
- You can only fill a shot glass if you are holding the shot glass, your other hand is empty, the shot glass is empty and clean.
- You can only refill a shot glass if you are holding the shot glass, your other hand is empty, the shot glass is empty and has contained the saree ingredient before.
- You can only empty a shot glass if you are holding the shot glass and it contains a beverage.
- You can only pour from a shot glass to a clean shaker if you are holding the shot glass, the shot glass contains an ingredient, and the shaker is empty and clean.
- You can only pour from a shot glass to a used shaker if you are holding the shot glass, the shot glass contains an ingredient, the shaker is unshaked and at a level not full.
- You can only empty a shaker if you are holding the shaker and the shaker contains a shaked beverage.
- You can only clean a shaker if you are holding the shaker, your other hand is empty, and the shaker is empty.
- You can only shake a cocktail if you are holding the shaker, your other hand is empty, the shaker is unshaked, and the shaker contains two ingredients, and both ingredients are parts of a cocktail.
- You can only pour from a shaker to a shot glass if you are holding the shaker, the shaker contains the cocktail, the shaker is shaked, and the shot glass is empty and clean.

- Once you grasp a container, you are holding the container and the container is not on the table.
- Once you leave a container on the table, your hand become empty.
- Once you pour an ingredient from a shot glass to a shaker, the shaker contains the ingredient and is at one level above the previous level, and the shot glass becomes empty.
- Once you empty a shaker, the shaker is at the empty level.
- Once you shake, the two ingredients in the shaker become a cocktail.
- Once you pour from a shaker to a shot glass, the shot glass contains the beverage in the shaker, the shot glass is no longer clean and empty, and the shaker is at one level below the previous level."""

gripper_instruction: str = """
You are a robot with a gripper that can move objects between different rooms. Your name is Robby.

## There are three actions defined in this domain:
- think xxx:
  - ✅ Format: `think: I need to move block A first because...`
  - ❌ Any other format is invalid and will not be recognized by the system.
- move <room1> <room2>: This action allows the robot to move from one room to another.The action has a single precondition, which is that the robot is currently in a room. The effect of this action is to move the robot to another room and to remove the fact that it is in the original room.
- pick <obj> <room> <gripper>: This action allows the robot to pick up an object using the gripper. The action has three preconditions: (1) the object is located in a room (2) the robot is currently in the same room and (3) the gripper is free (i.e., not holding any object). The effect of this action is to update the state of the world to show that the robot is carrying the object using the gripper, the object is no longer in the room, and the gripper is no longer free.
- drop <obj> <room> <gripper>: This action allows the robot to drop an object that it is carrying. The action has two preconditions: (1) the robot is currently carrying the object using the gripper, and (2) the robot is currently in a room. The effect of this action is to update the state of the world to show that the robot is no longer carrying the object using the gripper, the object is now located in the room, and the gripper is now free.

## NOTE:
- You should strictly follow these actions, no any other actions are allowed!
- Before you respond, you must ensure that your output actions strictly comply with the rules described above!
"""

tyreworld_instruction: str = """Your goal is to replace flat tyres with intact tyres on the hubs. Remember to open boot first to get tools you need. Intact tyres should be inflated. The nuts should be tight on the hubs. The flat tyres, wrench, jack, and pump should be in the boot. The boot should be closed.

## There are 13 actions defined in this domain:
- think xxx:
  - ✅ Format: `think: I need to move block A first because...`
  - ❌ Any other format is invalid and will not be recognized by the system.
- open <container>: The precondition for this action is that the container is unlocked and closed. The effect of this action is that the container is open and not closed.
- close <container>: The precondition for this action is that the container is open. The effect of this action is that the container is closed and not open.
- fetch <object> <container>: The precondition for this action is that the object is inside the container and the container is open. The effect of this action is that the object is held by the agent and not inside the container.
- put-away <object> <container>: The precondition for this action is that the object is held by the agent and the container is open. The effect of this action is that the object is inside the container and not held by the agent.
- loosen <nut> <hub>: The precondition for this action is that the agent has a wrench, the nut on hub is tight, and the hub is on the ground. The effect of this action is that the nut on hub is loose and not tight.
- tighten <nut> <hub>: The precondition for this action is that the agent has a wrench, the nut on hub is loose, and the hub is on the ground. The effect of this action is that the nut on hub is tight and not loose.
- jack-up <hub>: This action represents the process of lifting a hub off the ground using a jack. It requires the agent to have a jack and for the hub to be on the ground. After performing this action, the hub will no longer be on the ground and the agent will no longer have the jack.
- jack-down <hub>: This action represents the process of lowering a hub back to the ground from an elevated position using a jack. It requires the agent to have the hub off the ground. After performing this action, the hub will be back on the ground and the agent will have the jack.
- undo <nut> <hub>: This action undo the fastening of a nut on a hub. The preconditions are the hub is not on the ground (i.e., it has been jacked up), the hub is fastened, the agent has a wrench and the nut is loose. The effects are the agent has the nut, the hub is unfastened, the hub is no longer loose and the hub is not fastened anymore.
- do-up <nut> <hub>: This action fasten a nut on a hub. The preconditions are the agent has a wrench, the hub is unfastened, the hub is not on the ground (i.e., it has been jacked up) and the agent has the nut to be fastened. The effects are the nut is now loose on the hub, the hub is fastened, the hub is no longer unfastened and the agent no longer has the nut.
- remove-wheel <wheel> <hub>: This action removes a wheel from a hub. It can only be performed if the hub is not on the ground, the wheel is currently on the hub, and the hub is unfastened. After the action is performed, the agent will have the removed wheel and the hub will be free, meaning that the wheel is no longer on the hub.
- put-on-wheel <wheel> <hub>: This action puts a wheel onto a hub. It can only be performed if the agent has the wheel, the hub is free, the hub is unfastened, and the hub is not on the ground. After the action is performed, the wheel will be on the hub, the hub will no longer be free, and the agent will no longer have the wheel.
- inflate <wheel>: This action inflates a wheel using a pump. It can only be performed if the agent has a pump, the wheel is not inflated, and the wheel is intact. After the action is performed, the wheel will be inflated.

## NOTE:
- You should strictly follow these actions, no any other actions are allowed!
- Before you respond, you must ensure that your output actions strictly comply with the rules described above!
"""


pddl_prompts = {
    "blockworld": {
        "examples": [
            " Goal: The goal is to satisfy the following conditions: b1 is on b2., b2 is on b3.\nObservation: b1 is on the table.  b2 is on the table.  B3 is on the table. Robot arm is empty. The b1 is clear. The b2 is clear. The b3 is clear. \nAction: pickup b2\nObservation: b1 is on the table.  B2 is on the table.  The b1 is clear. The b3 is clear. You are holding b2.  \nAction: stack b2 b3\nObservation: b1 is on the table.  b1 is on b2. B3 is on the table. Robot arm is empty. The b1 is clear. The b2 is clear. \nAction: pickup b2. \nObservation: The action is not valid and therefore takes no effect. Please remember to satisfy the restriction of actions. You can also check valid actions. \n Action: check valid actions. \n Observation: valid actions are: pickup b2, unstack b1 b2.\nAction: pickup b1\nObservation: b2 is on b3. B3 is on the table.  Robot arm is empty. The b2 is clear.  You are holding b1. \nAction: stack b1 b2\nObservation: b1 is on b2. b2 is on b3. B3 is on the table.  Robot arm is empty. The b1 is clear. The goal is satisfied.\n"
        ],
        "instruction": blockworld_instruction,
        "system_msg": "You are a master in planning.",
    },
    "barman": {
        "examples": [
            "The goal is to satisfy the following conditions: shot1 contains cocktail1. \nObservation: Cocktail1 part1 ingredient is ingredient1. Cocktail1 part2 ingredient is ingredient3. Cocktail2 part1 ingredient is ingredient2. Cocktail2 part2 ingredient is ingredient3. Cocktail3 part1 ingredient is ingredient1. Cocktail3 part2 ingredient is ingredient2. Dispenser1 dispenses ingredient1. Dispenser2 dispenses ingredient2. Dispenser3 dispenses ingredient3. Left hand is empty. Level l0 is next to level l1. Level l1 is next to level l2. Right hand is empty. Shaker1 is at empty level l0. Shaker1 is at level l0. Shaker1 is clean. Shaker1 is empty. Shaker1 is on the table. Shot1 is clean. Shot1 is empty. Shot1 is on the table. Shot2 is clean. Shot2 is empty. Shot2 is on the table. Shot3 is clean. Shot3 is empty. Shot3 is on the table. Shot4 is clean. Shot4 is empty. Shot4 is on the table.\nAction: check valid actions\nObservation: Valid actions are: right grasp shot1. , left grasp shot3. , left grasp shot1. , left grasp shot2. , right grasp shaker1. , right grasp shot4. , right grasp shot3. , right grasp shot2. , left grasp shaker1. , left grasp shot4. , check valid actions\nAction: right grasp shot1.\nObservation: You are holding right.\nAction: fill-shot glass shot1 with ingredient3 with right and left holding dispenser3\nObservation: Shot1 contains ingredient3. \nAction: check valid actions\nObservation: Valid actions are: pour-shot-to-clean-shaker from a shot glass shot1 with ingredient3 to a clean shaker shaker1 with hand right from level l0 to level l1, left grasp shot3. , left grasp shot2. , use hand right to empty-shot glass shot1 with beverage ingredient3. , right leave shot1. , left grasp shaker1. , left grasp shot4. , check valid actions\nAction: pour-shot-to-clean-shaker from a shot glass shot1 with ingredient3 to a clean shaker shaker1 with hand right from level l0 to level l1\nObservation:  Pour shot1 from a shot glass to a used shaker ingredient3 Shaker1 contains ingredient3. \nAction: clean-shot glass shot1 with ingredient3 with hand right holding shot glass and left\nObservation:  Shot1 is clean. \nAction: fill-shot glass shot1 with ingredient1 with right and left holding dispenser1\nObservation: Shot1 contains ingredient1.\nAction: pour-shot-to-used-shaker from a shot glass shot1 with ingredient1 to a used shaker shaker1 with hand right from level l1 to level l2\nObservation: Pour shot1 from a shot glass to a used shaker ingredient1 Shaker1 contains ingredient1. Shaker1 contains ingredient3. Shaker1 is at empty level l0. Shaker1 is at level l2. Shaker1 is on the table. Shaker1 is unshaked.\nAction: clean-shot glass shot1 with ingredient1 with hand right holding shot glass and left\nObservation: Shot1 is clean. \nAction: right leave shot1\nObservation: Right hand is empty.\nAction: right grasp shaker1\nObservation: You are holding right.\nAction: shake a cocktail cocktail1 with ingredient ingredient1 and ingredient ingredient3 in a shaker shaker1 with hand right and hand left\nObservation: Shaker1 is shaked.\nAction: pour-shaker-to-shot to a shot glass cocktail1 the ingredient shot1 with hand right from shaker shaker1 from level l2 to level l1\nObservation: Shot1 contains cocktail1. The task is completed."
        ],
        "instruction": barman_instruction,
        "system_msg": "You are a master in planning.",
    },
    "gripper": {
        "examples": [
            "The goal is to satisfy the following conditions: ball1 is at roomb. , ball2 is at roomb. , ball3 is at roomb. \nObservation: Ball1 is a ball. Ball1 is at rooma. Ball2 is a ball. Ball2 is at rooma. Ball3 is a ball. Ball3 is at rooma. Left is a gripper. Left is free. Right is a gripper. Right is free. Robby is at rooma. Room rooma Room roomb\nAction: Pick up ball1 at rooma with arm right.\nObservation: Ball1 is a ball. Ball1 is carrying right. Ball2 is a ball. Ball2 is at rooma. Ball3 is a ball. Ball3 is at rooma.  Left is a gripper. Left is free. Right is a gripper. Robby is at rooma. Room rooma Room roomb\nAction: Pick up ball2 at rooma with arm left.\nObservation: Ball1 is a ball. Ball1 is carrying right. Ball2 is a ball. Ball2 is carrying left. Ball3 is a ball. Ball3 is at rooma. Left is a gripper. Right is a gripper. Robby is at rooma. Room rooma Room roomb\nAction: Move from rooma to roomb\nObservation: Ball1 is a ball. Ball1 is carrying right. Ball2 is a ball. Ball2 is carrying left. Ball3 is a ball. Ball3 is at rooma. Left is a gripper. Right is a gripper. Robby is at roomb. Room rooma Room roomb\nAction: drop ball1 at rooma with arm right\nObservation: The action is not valid and therefore takes no effect. Please remember to satisfy the restriction of actions. You can also check valid actions.\nAction: check valid actions\nObservation: Valid actions are: drop ball1 roomb right, drop ball2 roomb left, Move from roomb to rooma. , check valid actions\nAction: drop ball1 roomb right\nObservation: Ball1 is a ball. Ball1 is at roomb. Ball2 is a ball. Ball2 is carrying left. Ball3 is a ball. Ball3 is at rooma. Left is a gripper. Right is a gripper. Right is free. Robby is at roomb. Room rooma Room roomb\nAction: drop ball2 roomb left\nObservation: Ball1 is a ball. Ball1 is at roomb. Ball2 is a ball. Ball2 is at roomb. Ball3 is a ball. Ball3 is at rooma. Left is a gripper. Left is free. Right is a gripper. Right is free. Robby is at roomb. Room rooma Room roomb\nAction: move from roomb to rooma\nObservation: Ball1 is a ball. Ball1 is at roomb. Ball2 is a ball. Ball2 is at roomb. Ball3 is a ball. Ball3 is at rooma.Left is a gripper. Left is free. Right is a gripper. Right is free. Robby is at rooma. Room rooma Room roomb\nAction: Pick up ball3 at rooma with arm right.\nObservation: Ball1 is a ball. Ball1 is at roomb. Ball2 is a ball. Ball2 is at roomb. Ball3 is a ball. Ball3 is carrying right.  Left is a gripper. Left is free. Right is a gripper. Robby is at rooma. Room rooma Room roomb\nAction: Move from rooma to roomb\nObservation: Ball1 is a ball. Ball1 is at roomb. Ball2 is a ball. Ball2 is at roomb. Ball3 is a ball. Ball3 is carrying right. Ball4 is a ball. Ball4 is at rooma. Left is a gripper. Left is free. Right is a gripper. Robby is at roomb. Room rooma Room roomb\nAction: drop ball3 rooms right\nObservation: Ball1 is a ball. Ball1 is at roomb. Ball2 is a ball. Ball2 is at roomb. Ball3 is a ball. Ball3 is at roomb. Ball4 is a ball. Ball4 is at rooma. Left is a gripper. Left is free. Right is a gripper. Right is free. Robby is at roomb. Room rooma Room roomb. The task is completed."
        ],
        "instruction": gripper_instruction,
        "system_msg": "You are a master in moving objects.",
    },
    "tyreworld": {
        "examples": [
            "The goal is to satisfy the following conditions: w1 is in boot. (Note you need to open boot first so that you can extract tools from it.)\nObservation: Boot is closed. Boot is unlocked. Hub the-hub1 is fastened. Hub the-hub1 is on the ground. Jack is in boot. Pump is in boot. R1 is in boot. The nut nuts1 on the hub the-hub1 is tight. Wheel r1 is intact. Wheel r1 is not inflated. Wheel w1 is on hub the-hub1. Wrench is in boot.\nAction: check valid actions.\nObservation: Valid actions are: Open boot. Check valid actions. Look around.\nAction: Open boot.\nObservation: Boot is open. \nAction: fetch wrench from boot\nObservation: You have wrench.\nAction: check valid actions\nObservation: Valid actions are: Fetch pump from boot., Fetch r1 from boot., Put-away wrench in boot., Close boot., Fetch jack from boot., Loosen the nut nuts1 on the hub the-hub1., check valid actions\nAction: Loosen the nut nuts1 on the hub the-hub1.\nObservation: The nut nuts1 on the hub the-hub1 is loose. \nAction: fetch jack from boot\nObservation: You have jack. \nAction: jack-up the hub the-hub1\nObservation: Hub the-hub1 is not on the ground. \nAction: Undo the fastening of the nut nuts1 on the hub the-hub1.\nObservation: Hub the-hub1 is unfastened. \nAction: Remove-wheel w1 from the hub the-hub1.\nObservation: You have w1. \nAction: Put-away w1 in boot\nObservation: W1 is in boot. Goal is completed."
        ],
        "instruction": tyreworld_instruction,
        "system_msg": "You are a master in car repair.",
    },
}
