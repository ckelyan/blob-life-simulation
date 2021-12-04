import curses
import imageio
import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Simulate blob lives")
parser.add_argument("-s", "--set", help="Name of the parameter set")
parser.add_argument("-c", "--cmap", help="Colormap for the final plot", default="GnBu")

args = vars(parser.parse_args())

with open("parameters.yml", "r") as f:
    params = yaml.load(f, yaml.FullLoader)[0]

    if args["set"] in params.keys():
        params = params[args["set"]]

    else:
        raise KeyError(args['set'])

sexdict = {0: "f", 1: "m"}


class Blob:
    def __init__(self, x, y, grid_size):
        self.grid_size = grid_size
        self.x = x
        self.y = y
        self.energy = params["blob_init_energy"]
        self.dead = False
        self.sex = sexdict[np.random.binomial(1, params["mf_ratio"])]
        self.age = 0
        self.death_age = int(np.random.normal(
            params["expected_death"][self.sex], params["death_variance"]))

    def birth_relocate(self):
        if self.x < 0:
            self.x = 0
        elif self.x > self.grid_size-1:
            self.x = self.grid_size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.grid_size-1:
            self.y = self.grid_size-1

    def available_moves(self, coor):
        if (coor - params["max_travel"]) <= 0:
            available = (params["max_travel"]-coor, 2*params["max_travel"])
        elif (coor + params["max_travel"]) > (self.grid_size):
            available = (0, self.grid_size-1 - coor + params["max_travel"])
        else:
            available = (0, 2 * params["max_travel"])
        return available

    def move(self, changex, changey):
        moved = False
        if changex != 0:
            self.x += changex
            moved = True
        if changey != 0:
            self.y += changey
            moved = True

        if moved:
            self.energy += int(np.sqrt(changex**2 + changey**2)
                               * params["travel_energy"])

    def eat(self, foodpos):
        if ([self.x, self.y] in foodpos) and (not self.dead):
            self.energy += params["food_energy"]

    def live(self):
        self.age += 1
        self.energy += params["blob_live_energy"]
        if np.random.binomial(1, params["survival_chance"][self.sex]):
            if self.age >= self.death_age:
                self.dead = True
                return 1
            else:
                self.dead = False
                return 0
        else:
            self.dead = True
            return 1

    def death(self):
        if self.energy <= 0:
            self.dead = True
            return 1
        else:
            return 0


class Sim:
    def __init__(self):
        self.size = params["grid_size"]
        self.nblobs = int(self.size**2 * params["blobs_density"])
        self.nfood = int(self.nblobs * params["food_availability"])

        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.initgrid()
        self.day = 0

    def initgrid(self):
        self.blobpos = list(map(lambda x: [
                            x // self.size, x % self.size], np.random.randint(0, self.size**2, self.nblobs)))
        self.blobs = [Blob(*self.blobpos[i], self.size)
                      for i in range(self.nblobs)]
        self.foodpos = list(map(lambda x: [
                            x // self.size, x % self.size], np.random.randint(0, self.size**2, self.nfood)))

    def updatevalues(self):
        self.births = 0
        self.deaths = 0
        for blob in self.blobs:
            self.deaths += blob.live()
            blob.move(np.random.randint(*blob.available_moves(blob.x)) -
                      params["max_travel"], np.random.randint(*blob.available_moves(blob.y)) - params["max_travel"])

        self.blobs = [b for b in self.blobs if not b.dead]
        self.blobpos = list(map(lambda b: [b.x, b.y], self.blobs))

        for y in range(self.size):
            for x in range(self.size):
                if [x, y] in self.blobpos:
                    males = [blob for blob in self.blobs if (
                        (blob.x == x) and (blob.y == y) and (blob.sex == "m"))]
                    females = [blob for blob in self.blobs if (
                        (blob.x == x) and (blob.y == y) and (blob.sex == "f"))]
                    while len(males) > 1:
                        sorted_males = sorted(males, key=lambda x: x.energy)
                        weak_male = sorted_males[0]
                        winning_males = sorted_males[1:]
                        winning_males_energy = sum(
                            blob.energy for blob in winning_males)
                        weak_male.dead = True
                        added_energy = [weak_male.energy*params["death_energy_factor"]
                                        * blob.energy/winning_males_energy for blob in winning_males]
                        for i, blob in enumerate(winning_males):
                            blob.energy += int(added_energy[i])
                        males = winning_males
                        self.deaths += 1
                    if males:
                        num_children = 0
                        for female in females:
                            female.dead = np.random.binomial(
                                1, params["pregnancy_death_chance"])
                            num_children += np.random.geometric(
                                1-params["birth_chance"])
                            self.births += num_children
                        for k in range(num_children):
                            child = Blob(x+np.random.randint(-1, 1),
                                         y+np.random.randint(-1, 1), self.size)
                            child.birth_relocate()
                            self.blobs.append(child)
        self.blobs = [b for b in self.blobs if not b.dead]
        for blob in self.blobs:

            blob.eat(self.foodpos)
            self.deaths += blob.death()

        self.blobs = [b for b in self.blobs if not b.dead]

        self.blobpos = list(map(lambda b: [b.x, b.y], self.blobs))
        self.mpos = list(map(lambda b: [b.x, b.y] if (
            b.sex == "m") else None, self.blobs))
        self.mpos = list(filter(None, self.mpos))
        self.fpos = list(map(lambda b: [b.x, b.y] if (
            b.sex == "f") else None, self.blobs))
        self.fpos = list(filter(None, self.fpos))

        self.update_food()

    def update_food(self):
        self.foodpos = [food for food in list(
            self.foodpos) if food not in list(self.blobpos)]
        self.foodpos.extend(list(map(lambda x: [x // self.size, x % self.size], np.random.randint(
            0, self.size**2, int(len(self.foodpos)*params["food_regeneration"])))))
        self.foodpos = list(map(list, set(map(tuple, self.foodpos))))

    def applygrid(self, mval, fval, foodval):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        rows, cols = zip(*self.mpos)
        self.grid[rows, cols] = mval

        rows, cols = zip(*self.fpos)
        self.grid[rows, cols] = fval

        if len(self.foodpos) != 0:
            rows, cols = zip(*self.foodpos)
            self.grid[rows, cols] = foodval

    def plotgrid(self):
        total_energy = 0
        for i, blob in enumerate(self.blobs):
            total_energy += blob.energy
        total_energy /= len(self.blobs)
        
        stats = {
            "nblobs": len(self.blobs),
            "nmales": len(self.mpos),
            "nfemales": len(self.fpos),
            "nbirths": self.births,
            "ndeaths": self.deaths,
            "nfood": len(self.foodpos),
            "avgen": int(total_energy)
        }
        
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, interpolation="none", cmap=args["cmap"])
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Simluation day {self.day}")
        plt.gcf().text(0.02, 0.5,
f"Statistics\n\nBlobs: {stats['nblobs']}\n\n\
Males: {stats['nmales']}\n\n\
Females: {stats['nfemales']}\n\n\
Births: {stats['nbirths']}\n\n\
Deaths: {stats['ndeaths']}\n\n\
Food: {stats['nfood']}\n\n\
Average energy: ~{stats['avgen']}", 
                       fontsize=14)
        # for i, blob in enumerate(self.blobs):
        #    plt.text(blob.y,blob.x,f"{blob.energy}", color="grey", fontsize=6)
        # plt.savefig(f"day_{self.day:03d}.png")
        plt.subplots_adjust(left=0.3)
        plt.show()

    def nextframe(self):
        self.updatevalues()
        self.applygrid(3, 2, 1)
        self.plotgrid()
        self.day += 1


def main():
    s = Sim()
    for day in range(params["n_days"]):
        s.nextframe()


main()
