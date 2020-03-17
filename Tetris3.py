import numpy as np
import time
import random
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT

env = gym_tetris.make('TetrisA-v3')
env = JoypadSpace(env, MOVEMENT)
num_bins = 12
MOVELEFT = 6
MOVERIGHT = 3
MOVEDOWN = 9
RENDER = True


P = {
        0: { # I
            0: [(0,0), (1,0), (2,0), (3,0)],
            90: [(0,0), (0,1), (0,2), (0,3)],
            180: [(3,0), (2,0), (1,0), (0,0)],
            270: [(1,3), (1,2), (1,1), (1,0)],
        },
        1: { # T
            0: [(1,0), (0,1), (1,1), (2,1)],
            90: [(0,1), (1,2), (1,1), (1,0)],
            180: [(1,1), (2,0), (1,0), (0,0)],
            270: [(1,1), (0,0), (0,1), (0,2)],
        },
        2: { # J
            0: [(0,0), (0,1), (0,2), (1,2)],
            90: [(0,1), (1,1), (2,1), (2,0)],
            180: [(1,2), (1,1), (1,0), (0,0)],
            270: [(2,0), (1,0), (0,0), (0,1)],
        },
        3: { # L
            0: [(1,0), (1,1), (1,2), (0,2)],
            90: [(0,0), (1,0), (2,0), (2,1)],
            180: [(0,2), (0,1), (0,0), (1,0)],
            270: [(2,1), (1,1), (0,1), (0,0)],
        },
        4: { # S
            0: [(0,0), (1,0), (1,1), (2,1)],
            90: [(0,2), (0,1), (1,1), (1,0)],
            180: [(2,1), (1,1), (1,0), (0,0)],
            270: [(1,0), (1,1), (0,1), (0,2)],
        },
        5: { # Z
            0: [(2,0), (1,0), (1,1), (0,1)],
            90: [(0,0), (0,1), (1,1), (1,2)],
            180: [(0,1), (1,1), (1,0), (2,0)],
            270: [(1,2), (1,1), (0,1), (0,0)],
        },
        6: { # O
            0: [(1,0), (2,0), (1,1), (2,1)],
            90: [(1,0), (2,0), (1,1), (2,1)],
            180: [(1,0), (2,0), (1,1), (2,1)],
            270: [(1,0), (2,0), (1,1), (2,1)],
        }
    }


def FindMin(a , b, c):
    if a<b and a<c:
        return MOVERIGHT
    elif c<a and c<b:
            return MOVELEFT
    else:
        return MOVEDOWN

def GetHeight(gray_col,info):
	h = 0
	for j in range(20-info['board_height'],20):
		if gray_col[j]:
			h=20-j
			break
	return h

def AggHeight(state_gray):
	height = 0
	for i in range(10):
		height+= Height(state_gray[:,i])
	return height

def GetBumpiness(state_gray):
	bump = 0
	for i in range(9):
		bump+= abs(Height(state_gray[:,i])-Height(state_gray[:,i+1]))
	return bump

def GetHoles(state_gray):
	holes = 0
	for i in range(10):
		h = Height(state_gray[:,i])
		for j in range(20-h,20):
			if not state_gray[j,i]:
				holes+=1
	return holes

def lines_cleared(state):
    lines = 0
    for line in state:
        if min(line)==255:
            lines+=1
    return lines

def Height(gray_col):
    if max(gray_col)==0:
        return 0
    return len(gray_col)-np.argmax(gray_col)

def width(piece):
	w = max([p[0] for p in piece])
	return w+1

def GetPieceId(peice):
    if peice[0]=='I':
        return 0
    elif peice[0]=='T':
        return 1
    elif peice[0]=='J':
        return 2
    elif peice[0]=='L':
        return 3
    elif peice[0]=='S':
        return 4
    elif peice[0]=='Z':
        return 5
    elif peice[0]=='O':
        return 6

def PieceId(piece_id, rotation):
	if piece_id == 6:
		return 'O'
	elif piece_id == 5:
		if rotation == 0 or rotation == 180:
			return 'Zh'
		elif rotation == 90 or rotation == 270:
			return 'Zv'

	elif piece_id == 4:
		if rotation == 0 or rotation == 180:
			return 'Sh'
		elif rotation == 90 or rotation == 270:
			return 'Sv'

	elif piece_id == 0:
		if rotation == 0 or rotation == 180:
			return 'Ih'
		elif rotation == 90 or rotation == 270:
			return 'Iv'

	elif piece_id == 1:
		if rotation == 0:
			return 'Td'
		elif rotation == 90:
			return 'Tl'
		elif rotation == 180:
			return 'Tu'
		elif rotation == 270:
			return 'Tr'

	elif piece_id == 2:
		if rotation == 0:
			return 'Jr'
		elif rotation == 90:
			return 'Jd'
		elif rotation == 180:
			return 'Jl'
		elif rotation == 270:
			return 'Ju'

	elif piece_id == 3:
		if rotation == 0:
			return 'Ll'
		elif rotation == 90:
			return 'Lu'
		elif rotation == 180:
			return 'Lr'
		elif rotation == 270:
			return 'Ld'

def next_states(current_state, info):
    next_states =[]
    next_piece_ids = []
    next_rotations = []
    piece_id = GetPieceId(info['current_piece'])
    #print(info['current_piece'][0])
    if piece_id == 6:
        rotations = [0]
    elif piece_id == 0 or piece_id == 4 or piece_id == 5:
        rotations = [0,90]
    else:
        rotations = [0,90,180,270]
        
    for rotation in rotations:
        piece = P[piece_id][rotation]
        min_x = min([p[0] for p in piece])
        max_x = max([p[0] for p in piece])
        min_y = min([p[1] for p in piece])
        max_y = max([p[1] for p in piece])
        board_width = 10

        for x in range(0, board_width - max_x):
            isInvalid = False
            state = np.copy(current_state)
            for i in range(3):
                for j in range(len(state[0])):
                    if state[i][j]>0:
                        state[i][j] = 0
                
            heights = []
            for i in range(0,max_x+1):
                heights.append(Height(state[:,x+i]))

            index_H = np.argmax(heights)
            max_H = max(heights)
            max_H_col = x + index_H
            same_height_pieces = []
            ys_of_same_height = []
            if heights.count(max_H)>1:
                for i in range(len(heights)):
                    if heights[i] == max_H:
                        for p in piece:
                            if p[0] == i:
                                same_height_pieces.append(p)
                                ys_of_same_height.append(p[1])
                origin = same_height_pieces[np.argmin(ys_of_same_height)]
                    
            else:
                a = []
                for c in piece:
                    if c[0]==index_H:
                        a.append(c[1])
                b = a.copy()
                b.clear()
                #if a == b:
                    #print("a is EMpty")
                try:
                    origin = (index_H ,min(a))
                except:
                    origin = (index_H ,min_y)
             
            origin = np.array(origin)
            for k in piece:
                X = k[0] - origin[0]
                Y = k[1] - origin[1]
                if 19-max_H - Y <0 or max_H_col + X < 0 :
                    isInvalid = True
                    break
                
                try:
                    if state[19-max_H - Y][max_H_col + X]==0:
                        state[19-max_H - Y][max_H_col + X] += 255
                        
                    elif state[19-max_H - Y][max_H_col + X]== 255 :
                        isInvalid = True
                        break
                    else:
                        print(state[19-max_H - Y][max_H_col + X])
                        
                except:
                    isInvalid = True
                    break
            
            if not isInvalid:
                next_states.append(state)
                next_piece_ids.append(piece_id)
                next_rotations.append(rotation)
    return next_states, next_piece_ids, next_rotations

def best_state_index(next_states):
    values = []
    for state in next_states:
        height = AggHeight(state)
        bump = GetBumpiness(state)
        holes = GetHoles(state)
        lines = lines_cleared(state)
        values.append(-0.51*height-0.18*bump-0.36*holes+7.6*lines)
    best_index = np.argmax(values)
    best = next_states[best_index]
    return best_index

def TakeBestAction(optimal, gray, info):
    optim_x = 99
    curr_x = 98
    for i in range(20):
        if np.max(optimal[i]) == 255:
            optim_x = np.argmax(optimal[i])
            break

    for i in range(20-info['board_height']):
        if np.max(gray[i]) == 255:
            curr_x = np.argmax(gray[i])
            break
    return optim_x, curr_x
    
    
    
EPISODES = 1000
n=0
scores = []
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    episode_reward = 0
    thresh = 7
    step = 1
    i=0
    current_state = env.reset()
    curr_info = {'board_height': 0, 'statistics': {'Z': 0, 'O': 0, 'J': 0, 'T': 0, 'I': 0, 'S': 0, 'L': 0}, 'score': 0, 'current_piece': '', 'next_piece': '', 'number_of_lines': 0}
    
    current_state = current_state[47:205,95:176]
    current_state = cv2.resize(current_state,None,fy=0.12345679, fx=2*0.0632911, interpolation = cv2.INTER_CUBIC)
    current_gray = cv2.cvtColor(current_state, cv2.COLOR_BGR2GRAY)
    r, current_gray = cv2.threshold(current_gray,1,255,cv2.THRESH_BINARY)
    done = False
    best_piece = None
    while not done:
        new_state , reward, done, info = env.step(env.action_space.sample())
        #print("Random action")
        if RENDER:
            env.render()
        new_state = new_state[47:205,95:176]
        new_state = cv2.resize(new_state,None,fy=0.12345679, fx=2*0.0632911, interpolation = cv2.INTER_CUBIC)
        new_gray = cv2.cvtColor(new_state, cv2.COLOR_BGR2GRAY)
        r, new_gray = cv2.threshold(new_gray,1,255,cv2.THRESH_BINARY)
        
        if info['statistics'] != curr_info['statistics'] or info['board_height']!= curr_info['board_height'] or info['current_piece'] == None:
                if info['current_piece'] == None:
                    while(info['current_piece'] == None) :
                        new_state , reward, done, info = env.step(env.action_space.sample())
                        #print("Random action")
                        if RENDER:
                            env.render()
                        new_state = new_state[47:205,95:176]
                        new_state = cv2.resize(new_state,None,fy=0.12345679, fx=2*0.0632911, interpolation = cv2.INTER_CUBIC)
                        new_gray = cv2.cvtColor(new_state, cv2.COLOR_BGR2GRAY)
                        r, new_gray = cv2.threshold(new_gray,1,255,cv2.THRESH_BINARY)
                nex, next_piece_ids, next_rotations = next_states(new_gray, info)
                best_index = best_state_index(nex)
                best_state = nex[best_index]
                best_piece_id = next_piece_ids[best_index]
                best_rotation = next_rotations[best_index]
                best_piece = PieceId(best_piece_id, best_rotation)
                #print("BEST PIECE UPDATED===============================")
                curr_info = info
        while best_piece != info['current_piece'] and not done:
                new_state, reward, done, info = env.step(env.action_space.sample())
                if RENDER:
                        env.render()
                #print("Gettin There")
                if done or best_piece == info['current_piece']:
                        break
                current_state = new_state
        k = 0
        while best_piece == info['current_piece'] and not done:
                try:
                        state = new_state[47:205,95:176]
                        state = cv2.resize(state,None,fy=0.12345679, fx=2*0.0632911, interpolation = cv2.INTER_CUBIC)
                except:
                        state = new_state
                gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
                r, gray = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
                optim = best_state - gray
                r, optim = cv2.threshold(optim,1,255,cv2.THRESH_BINARY)
                current_gray = np.copy(gray)
                for i in range(20):
                    if i>=20-info['board_height']:
                        for j in range(10):
                            if current_gray[i][j] >= 255:
                                current_gray[i][j] = 0
                                
                optim_x, curr_x = TakeBestAction(optim ,current_gray, info)
                if curr_x == optim_x:
                    action =  MOVEDOWN
                    n+=1
                elif curr_x < optim_x:
                    action = MOVERIGHT
                elif curr_x > optim_x:
                    action = MOVELEFT
                    
                if action == MOVEDOWN and n> thresh:
                    k = info['statistics']
                    n=0
                    if info['board_height']>15:
                        thresh*=0.95
                    while(k == info['statistics'] and curr_x == optim_x ):
                        new_state, reward, done, info = env.step(action)
                        if RENDER:
                            env.render()
                        if done:
                            break
                else:      
                    new_state, reward, done, info = env.step(action)
                    if RENDER:
                        env.render()
        current_gray = new_gray
        current_state = new_state
        if done:
            scores.append(info['score'])
            print("Game Finished with Score = ", info['score'])
            print("Cleared lines = ", info['number_of_lines'])
            print("Thresh=", thresh)
            if info['number_of_lines'] > 50:
                cv2.imwrite('Game{}.png'.format(episode),current_state)
            env.reset()
