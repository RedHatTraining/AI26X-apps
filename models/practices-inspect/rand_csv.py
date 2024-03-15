import string, random
from random import randint


def rand_initials():
    return "".join(random.choice(string.ascii_uppercase) for i in range(3))


print("Initials,Age,Weight")
for i in range(10000):
    print(f"{rand_initials()},{randint(10, 90)},{randint(100, 200)}")
