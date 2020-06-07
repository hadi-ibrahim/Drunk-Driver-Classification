L = [0,1,2,3,4]
for x in range (len(L)//2):
    L[x] ,L[len(L)-x-1] = L[len(L)-x-1] ,L[x]
print (L)