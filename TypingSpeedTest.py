from collections import OrderedDict
from operator import itemgetter    
import time
import random
Dic = {}
List = []
falsecount=0
turecount=0
TotalWords=0

print ("Please Enter you name: ")
Name = input()
Name = Name + " "

ReadFile = open("E:\MyFileMyChoise.txt","r+")
if Name not in ReadFile.read():
   ReadFile.write(Name)
   ReadFile.write(": 10")
   ReadFile.write("\n")
   ReadFile.close()

print("Welcome",Name)

ReadFile = open("E:\MyFileMyChoise.txt","r+")

for Line in ReadFile.readlines():
    Dic[Line[0: -5]] = int(Line[-3:-1])


Dic = sorted(Dic.items(), key = itemgetter(1), reverse = True)

Dictionary = open("E:\Dictionary.txt")
i = 0
for x in Dictionary.readlines():
    List+=x

input()
timeend = int(time.clock() + 120)

while time.clock() < timeend:
    Test = List[random.randint(0,i)]
    print(Test)
    if input() is Test:
        print(True) #Green krna ha
        turecount +=1
    else:
        print(False) #Red krna ha
        falsecount +=1
    TotalWords +=1

truecount/2