from envs.grids.game_objects import Actions
import random, math, os
import numpy as np


class MordorWorld:

    def __init__(self):
        self._load_map()
        self.map_height, self.map_width = 12,9

    def reset(self):
        self.agent = (1,1)

    def execute_action(self, a):
        """
        We execute 'action' in the game
        """
        x,y = self.agent
        self.agent = self._get_new_position(x,y,a)

    def _get_new_position(self, x, y, a):
        action = Actions(a)
        # executing action
        if (x,y,action) not in self.forbidden_transitions:
            if action == Actions.up   : y+=1
            if action == Actions.down : y-=1
            if action == Actions.left : x-=1
            if action == Actions.right: x+=1
        return x,y


    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = ""
        if self.agent in self.objects:
            ret += self.objects[self.agent]
        return ret

    def get_features(self):
        """
        Returns the features of the current state (i.e., the location of the agent)
        """
        x,y = self.agent
        return np.array([x,y])

    def show(self):
        for y in range(8,-1,-1):
            if y % 3 == 2:
                for x in range(12):
                    if x % 3 == 0:
                        print("_",end="")
                        if 0 < x < 11:
                            print("_",end="")
                    if (x,y,Actions.up) in self.forbidden_transitions:
                        print("_",end="")
                    else:
                        print(" ",end="")
                print()                
            for x in range(12):
                if (x,y,Actions.left) in self.forbidden_transitions:
                    print("|",end="")
                elif x % 3 == 0:
                    print(" ",end="")
                if (x,y) == self.agent:
                    print("A",end="")
                elif (x,y) in self.objects:
                    print(self.objects[(x,y)],end="")
                else:
                    print(" ",end="")
                if (x,y,Actions.right) in self.forbidden_transitions:
                    print("|",end="")
                elif x % 3 == 2:
                    print(" ",end="")
            print()      
            if y % 3 == 0:      
                for x in range(12):
                    if x % 3 == 0:
                        print("_",end="")
                        if 0 < x < 11:
                            print("_",end="")
                    if (x,y,Actions.down) in self.forbidden_transitions:
                        print("_",end="")
                    else:
                        print(" ",end="")
                print()                

    def get_model(self):
        """
        This method returns a model of the environment. 
        We use the model to compute optimal policies using value iteration.
        The optimal policies are used to set the average reward per step of each task to 1.
        """
        S = [(x,y) for x in range(12) for y in range(9)] # States
        A = self.actions.copy() # Actions
        L = self.objects.copy() # Labeling function
        T = {}                  # Transitions (s,a) -> s' (they are deterministic)
        for s in S:
            x,y = s
            for a in A:
                T[(s,a)] = self._get_new_position(x,y,a)
        return S,A,L,T # SALT xD

    def _load_map(self):
        # Creating the map
        self.objects = {}
        self.objects[(0,0)] = "e"  # EXIT
        self.objects[(2,3)] = "k"  # KEY
        self.objects[(4,3)] = "v"  # VOLCANO
        self.objects[(10,0)] = "s" # SAM
        self.objects[(7,5)] = "r"  # RING
        self.objects[(4,0)] = "o"  # ORC
        self.objects[(8,0)] = "o"  # ORC
        self.objects[(6,1)] = "o"  # ORC
        self.objects[(3,2)] = "o"  # ORC
        self.objects[(8,3)] = "o"  # ORC
        self.objects[(2,4)] = "o"  # ORC
        self.objects[(5,4)] = "o"  # ORC
        self.objects[(9,4)] = "o"  # ORC
        self.objects[(4,5)] = "o"  # ORC
        self.objects[(2,7)] = "o"  # ORC
        self.objects[(6,7)] = "o"  # ORC
        self.objects[(11,8)] = "o" # ORC
        
        # Adding walls
        self.forbidden_transitions = set()
        # Add 4 long horizontal walls 
        for x in range(12):
            for y in [0,3,6]:
                self.forbidden_transitions.add((x,y,Actions.down)) 
                self.forbidden_transitions.add((x,y+2,Actions.up))
        
        # Add edge vertical walls
        for y in range(9):
            self.forbidden_transitions.add((0,y,Actions.left))
            self.forbidden_transitions.add((11,y,Actions.right))

        
        # Adding 'doors' for long horizontal walls
        for x in [1, 10]:
            self.forbidden_transitions.remove((x,2,Actions.up))
            self.forbidden_transitions.remove((x,3,Actions.down))
        for x in [1, 3, 5, 6, 8, 10]:
            self.forbidden_transitions.remove((x,5,Actions.up))
            self.forbidden_transitions.remove((x,6,Actions.down))
        
        # Adding the rest of horizontal walls
        for y in [0, 1]:
            self.forbidden_transitions.add((11,y,Actions.up))
            self.forbidden_transitions.add((11,y+1,Actions.down))

        self.forbidden_transitions.add((9,7,Actions.up))
        self.forbidden_transitions.add((9,8,Actions.down))
        self.forbidden_transitions.add((8,3,Actions.up))
        self.forbidden_transitions.add((8,4,Actions.down))
        self.forbidden_transitions.add((7,4,Actions.up))
        self.forbidden_transitions.add((7,5,Actions.down))
        self.forbidden_transitions.add((4,4,Actions.up))
        self.forbidden_transitions.add((4,5,Actions.down))
        self.forbidden_transitions.add((3,3,Actions.up))
        self.forbidden_transitions.add((3,4,Actions.down))
        self.forbidden_transitions.add((1,7,Actions.up))
        self.forbidden_transitions.add((1,8,Actions.down))
        self.forbidden_transitions.add((0,1,Actions.up))
        self.forbidden_transitions.add((0,2,Actions.down))

        # Adding the rest of vertical walls
        for x in [0, 2, 8, 9, 10]:
            self.forbidden_transitions.add((x,0,Actions.up))
            self.forbidden_transitions.add((x+1,0,Actions.down))
        for x in [2, 8]:
            self.forbidden_transitions.add((x,2,Actions.up))
            self.forbidden_transitions.add((x+1,2,Actions.down))
            self.forbidden_transitions.add((x,3,Actions.up))
            self.forbidden_transitions.add((x+1,3,Actions.down))
        for x in [3, 7]:
            self.forbidden_transitions.add((x,4,Actions.up))
            self.forbidden_transitions.add((x+1,4,Actions.down))
        for x in [4, 6]:
            self.forbidden_transitions.add((x,5,Actions.up))
            self.forbidden_transitions.add((x+1,5,Actions.down))
        for x in [0, 3, 7, 10]:
            self.forbidden_transitions.add((x,6,Actions.up))
            self.forbidden_transitions.add((x+1,6,Actions.down))
        for x in [0, 3, 7, 9]:
            self.forbidden_transitions.add((x,8,Actions.up))
            self.forbidden_transitions.add((x+1,8,Actions.down))

        # Adding the agent
        self.actions = [Actions.up.value,Actions.right.value,Actions.down.value,Actions.left.value]
