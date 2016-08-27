#!/usr/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.color import rgb2gray
import os
from skimage import io
import pickle as pkl
from math import ceil
# import bpy

def bits(name1, name2):
	print("Loading")
	img1 = io.imread("/Users/shira/Documents/FirstDegree/FirstSteps/DoubleOrient/Input/" + name1)
	img1 = rgb2gray(img1)

	img2 = io.imread("/Users/shira/Documents/FirstDegree/FirstSteps/DoubleOrient/Input/" + name2)
	img2 = rgb2gray(img2)

	N = len(img1)
	large1 = (img1[0][0] == 255)
	large2 = (img2[0][0] == 255)
	for i in range(N):
		for j in range(N):
			if large1:
				img1[i][j] = img1[i][j] / 255
			if large2:
				img2[i][j] = img2[i][j] / 255
			img1[i][j] = 1 if (img1[i][j] < 1) else 0
			img2[i][j] = 1 if (img2[i][j] < 1) else 0

	return img1[::-1], img2[::-1]


def align(img1, img2):
	print("Aligning")
	N = len(img1)
	i = 0
	h1 = N; h2 = N
	while (i < N) and not sum(img1[i]):
		i += 1
	if i > 0 and i < N-1:
		it = i
		while it < N and sum(img1[it]):
			for j in range(N):
				img1[it-i][j] = img1[it][j]
				img1[it][j] = 0
			it += 1
		h1 = it - i

	i = 0
	while (i < N) and not sum(img2[i]):
		i += 1
	if i > 0 and i < N-1:
		it = i
		while it < N and sum(img2[it]):
			for j in range(N):
				img2[it-i][j] = img2[it][j]
				img2[it][j] = 0
			it += 1
		h2 = it - i

	if (h1 != h2):
		if (h2 > h1):
			tmp = img1
			img1 = img2
			img2 = tmp
			jump = ceil(float(h2) / (h2 - h1))
		else:
			jump = ceil(float(h1) / (h1 - h2))
		shrink(img1,jump)


def _shrink(img, jump):
	N = len(img)
	it = 0
	for i in range(N-1):
		if i % jump != 0:
			for j in range(N):
				img[it][j] = img[i+1][j]
				if img[i][j] and (not it or not img[it-1][j]):
					img[it][j] = 1
			it += 1

	while (it < N):
		for j in range(N):
			img[it][j] = 0
		it += 1


def curv(img):
	print("Curving")
	N = len(img)
	curv = []
	count = -1
	for i in range(N):
		for j in range(N):
			if img[i][j]:
				count += 1
				if not curv:
					radius = neighbours(img, i, j)
					if len(radius) == 1:
						curv.append((i,j))
						curv.append(radius[0])
	while count:
		pi, pj = curv[-1][0], curv[-1][1]
		di, dj = pi-curv[-2][0], pj-curv[-2][1]
		vi = (pi + di >= 0 and pi + di < N)
		vj = (pj + dj >= 0 and pj + dj < N)
		
		n1 = neighbours(img, pi+di, pj) 
		n2 = neighbours(img, pi, pj+dj)
		n3 = neighbours(img, pi+di, pj+dj)
		if di != 0 and vi and img[pi + di][pj] and ((pi+di, pj) not in curv or len(n1) > 2):
			curv.append((pi+di, pj))
		elif dj != 0 and vj and img[pi][pj+dj] and ((pi, pj+dj) not in curv or len(n2) > 2):
			curv.append((pi, pj+dj))
		elif vi and vj and img[pi+di][pj+dj] and ((pi+di,pj+dj) not in curv or len(n3) > 2):
			curv.append((pi+di,pj+dj))
		elif di == 0 and pi < N-1 and img[pi+1][pj]:
			curv.append((pi+1, pj))
		elif di == 0 and pi > 0 and img[pi-1][pj]:
			curv.append((pi-1, pj))
		elif dj == 0 and pj < N-1 and img[pi][pj+1]:
			curv.append((pi, pj+1))
		elif dj == 0 and pj > 0 and img[pi][pj-1]:
			curv.append((pi, pj-1))
		elif di == 0 and pi < N-1 and vj and img[pi+1][pj+dj]:
			curv.append((pi+1, pj+dj))
		elif di == 0 and pi > 0 and vj and img[pi-1][pj+dj]:
			curv.append((pi-1, pj+dj))
		elif dj == 0 and pj < N-1 and vi and img[pi+di][pj+1]:
			curv.append((pi+di, pj+1))
		elif dj == 0 and pj > 0 and vi and img[pi+di][pj-1]:
			curv.append((pi+di, pj-1))
		elif dj != 0 and di != 0 and img[pi-di][pj]:
			curv.append((pi-di, pj))
		elif di != 0 and dj != 0 and img[pi][pj-dj]:
			curv.append((pi, pj-dj))
		elif di != 0 and vi and img[pi+di][pj-dj]:
			curv.append((pi+di,pj-dj))
		elif dj != 0 and vj and img[pi-di][pj+dj]:
			curv.append((pi-di, pj+dj))
		elif di == 0 and pi < N-1 and img[pi+1][pj-dj]:
			curv.append((pi+1, pj-dj))
		elif di == 0 and pi > 0 and img[pi-1][pj-dj]:
			curv.append((pi-1, pj-dj))
		elif dj == 0 and pj < N-1 and img[pi-di][pj+1]:
			curv.append((pi-di, pj+1))
		elif dj == 0 and pj > 0 and img[pi-di][pj-1]:
			curv.append((pi-di, pj-1))
		count -= 1
	return curv


def _neighbours(img, i, j):
	N = len(img)
	neighbours = []
	for it in range(i-1,min(i+2, N)):
		for jt in range(max(j-1,0),min(j+2, N)):
			if (it != i or jt != j) and img[it][jt]:
				neighbours.append((it,jt))
	return neighbours


def curv_path(img):
	N = len(img)
	start, end = None, None
	for i in range(N):
		for j in range(N):
			if img[i][j]:
				n = neighbours(img, i, j)
				if len(n) == 1:
					if not start:
						start = (i,j)
					else:
						end = (i,j)
	parent = {}
	used = [start]
	q = [start]
	while q:
		curr = q.pop()
		curr_neighbours = neighbours(img, curr[0], curr[1])
		for n in curr_neighbours:
			if n not in used:
				used.append(n)
				q.insert(0, n)
				parent[n] = curr

	curv = []
	if end in parent:
		while end != start:
			curv.append(end)
			end = parent[end]
	curv.reverse()
	return curv


def cylinder(img):
	print("Smudging")
	N = len(img1)
	points = []
	for i in range(N):
		for j in range(N):
			if img[i][j]:
				k = 0
				while (k < N):
					if k % 20 == 0:
						points.append((i,j,k))
					k += 1
	return points


def intersect(img1, img2):
	print("Intersecting")
	N = len(img1)
	cube_all = np.zeros((N,N,N))
	points = []
	for i in range(N):
		for j in range(N):
			for k in range(N):
				if img1[i][j] and img2[i][k]:
					cube_all[i][j][k] = 1
					points.append((i,j,k))
	starters = []
	for p in points:
		if len(_neighbours3d(cube_all, p[0], p[1], p[2])) == 1:
			starters.append(p)
	cmax = []
	for p in starters:
		cc = _select_cc(cube_all, p)
		if len(cc) > len(cmax):
			cmax = cc

	cube = np.zeros((N,N,N))
	for p in cmax:
		cube[p[0]][p[1]][p[2]] = 1
	return cube


def _select_cc(cube, start):
	used = [start]
	q = [start]
	while q:
		curr = q.pop()
		curr_neighbours = _neighbours3d(cube, curr[0], curr[1], curr[2])
		for n in curr_neighbours:
			if n not in used:
				used.append(n)
				q.insert(0, n)
	return used


def select(curv1, curv2, img1, img2, cube_all):
	print("Selecting")
	N = len(img1)
	cube = np.zeros((N,N,N))
	points = []
	p = 0
	prev = None
	while p < len(curv1):
		i, j = curv1[p][0], curv1[p][1]
		k = 0
		while k < N and not cube[i][j][k]:
			k += 1
		if k == N:
			k = prev[2] if prev else 0
			kadd, ksub = k, k
			while kadd < N and not img2[i][kadd]:
				kadd += 1
			while ksub >= 0 and not img2[i][ksub]:
				ksub -= 1

			if ksub < 0 and kadd == N:
				k = prev[2]
			elif ksub < 0 or abs(kadd - prev[2]) <= 1:
				k = kadd
			elif kadd == N or abs(ksub - prev[2]) <= 1:
				k = ksub
			elif kadd - ksub <= 3 and neighbours(img2, prev[0], prev[2]) > 2:
				k = kadd if not sum([col[kadd] for col in cube[i]]) else ksub
			# elif select_gap((i,k),(i,kadd),curv2) >= select_gap((i,k),(i,ksub),curv2):
			elif k-ksub < kadd-k:
				k = ksub
			else:
				k = kadd
			
			if prev and abs(k - prev[2]) > 1:
				if abs(k - prev[2]) > 1:
					start = (i, j, prev[2])
					end = (i, j, k)
					points.extend(_select_path(cube, cube_all, start, end))
					prev = (i, j, points[-1][2])
				if abs(k - prev[2]) > 1:
					print(k - prev[2])
				
		cube[i][j][k] = 1
		prev = (i, j, k)
		points.append(prev)
		p += 1

	prev = None
	p = 0
	while p < len(curv2):
		i, k = curv2[p][0], curv2[p][1]
		j = 0
		while j < N and not cube[i][j][k]:
			j += 1
		if j == N:
			j = prev[1] if prev else 0
			jadd, jsub = j, j
			while jadd < N and not img1[i][jadd]:
				jadd += 1
			while jsub >= 0 and not img1[i][jsub]:
				jsub -= 1

			if jsub >= 0 or jadd < N:
				j = prev[1]
			elif jsub < 0 or abs(jadd - prev[1]) <= 1:
				j = jadd
			elif jadd == N or abs(jsub - prev[1]) <= 1:
				j = jsub
			elif jadd - jsub <= 3 and neighbours(img1, prev[0], prev[1]) > 2:
				j = jadd if sum(cube[i][jadd]) else jsub
			# elif select_gap((i,j),(i,jadd),curv1) >= select_gap((i,j),(i,jsub),curv1):
			elif j-jsub < jadd-j:
				j = jsub
			else:
				j = jadd
			
			if prev and abs(j - prev[1]) > 1:
				if abs(j - prev[1]) > 1:
					start = (i, prev[1], k)
					end = (i, j, k)
					points.extend(_select_path(cube, cube_all, start, end))
					prev = (i, points[-1][1], k)
				if abs(j - prev[1]) > 1:
					print(j-prev[1])

		cube[i][j][k] = 1
		prev = (i, j, k)
		points.append(prev)
		p += 1

	return points


def _neighbours3d(cube, i, j, k):
	N = len(cube)
	n = []
	for it in range(max(i-1, 0), min(i+2, N)):
			for jt in range(max(j-1, 0), min(j+2, N)):
				for kt in range(max(k-1,0), min(k+2, N)):
					if cube[it][jt][kt] and (it,jt,kt) != (i,j,k):
						n.append((it,jt,kt))
	return n


def _select_path(cube, cube_all, start, end):
	parent = {}
	used = [start]
	q = [start]
	while q:
		curr = q.pop()
		curr_neighbours = _neighbours3d(cube_all, curr[0], curr[1], curr[2])
		for n in curr_neighbours:
			if n not in used:
				used.append(n)
				q.insert(0, n)
				parent[n] = curr
			if n == end:
				q = []
	
	points = []
	if end in parent:
		while end != start:
			points.append(end)
			cube[end[0]][end[1]][end[2]] = 1
			end = parent[end]
	points.reverse()
	return points


def plot_bits(img1, img2):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	N = len(img1)
	x, y, z = [], [], []
	for i in range(N):
		for j in range(N):
			if img1[i][j]:
				x.append(j); y.append(0); z.append(i)
			if img2[i][j]:
				x.append(0); y.append(j); z.append(i)
	ax.scatter(x, y, z, s=1)
	plt.show()


def plot_curv(curv1, curv2):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x1, y1, z1 = [], [], []
	for i in range(len(curv1)):
		x1.append(curv1[i][1]); y1.append(0); z1.append(curv1[i][0])
	
	x2, y2, z2 = [], [], []
	for i in range(len(curv2)):
		x2.append(0); y2.append(curv2[i][1]); z2.append(curv2[i][0])

	ax.plot(x1, y1, z1)
	ax.plot(x2, y2, z2)
	plt.show()


def plot_cube(cube):
	N = len(cube)
	x, y, z = [], [], []
	for i in range(N):
		for j in range(N):
			for k in range(N):
				if cube[i][j][k]:
					x.append(k);y.append(j);z.append(i)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x,y,z,s=1)
	plt.show()


def plot_cyls(cyl1, cyl2):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	
	x, y, z = [], [], []
	for i in range(len(cyl1)):
		x.append(cyl1[i][1]); y.append(cyl1[i][2]); z.append(cyl1[i][0])
	ax.scatter(x, y, z, s=10, c='r')

	x, y, z = [], [], []
	for i in range(len(cyl2)):
		x.append(cyl2[i][2]); y.append(cyl2[i][1]); z.append(cyl2[i][0])
	
	ax.scatter(x, y, z, s=10, c='g')
	plt.show()


def plot_sculp(points):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x, y, z = [], [], []
	for i in range(len(points)):
		x.append(points[i][1]); y.append(points[i][2]); z.append(points[i][0])
	
	ax.scatter(x, y, z, s=1)
	plt.show()


def main_dump(f1, f2):
	img1, img2 = bits(f1,f2)
	align(img1, img2)
	curv1 = curv(img1)
	curv2 = curv(img2)

	cube_all = intersect(img1, img2)
	points = select(curv1, curv2, img1, img2, cube_all)
	
	n1 = f1.split(".")[0]; n2 = f2.split(".")[0]
	out = open(n1 + "_" + n2 + ".pkl", "wb")
	pkl.dump(points, out)
	out.close()


def main_load(f1, f2):
	n1 = f1.split(".")[0]; n2 = f2.split(".")[0]
	inn = open(n1 + "_" + n2 + ".pkl", "rb")
	points = pkl.load(inn)
	plot_sculp(points)


# main_load("seal.png", "whale.png")
# main_load("rooster.png", "ostrich.png")
# main_load("deer.png", "kangaroo.png")
# main_load("fish.png", "crab.png")
