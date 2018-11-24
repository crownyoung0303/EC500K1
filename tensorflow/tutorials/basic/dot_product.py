def dot_product(length, vec1, vec2):
    """Returns the dot product of two vectors
    
    `vec1` and `vec2` are each `length` elements long
    """
    product = 0
    for i in range(length):
        product += vec1[i] * vec2[i]
    return product
    
def tester(tests):
        for i in range(len(tests)):
            outcome = " FAILED"
            this_test = tests[i]
            if this_test():
                outcome = " PASSED"
            print "Test ", i, outcome

def test1():
    a = [0, 1, 2]
    b = [1, 2, 1]
    a_dot_b = dot_product(len(a), a, b)
    if a_dot_b == 4:
        return True
    return False

def test2():    
    if dot_product(2, [4, 2], [1, 2]) == 8:
        return True
    return False
    
def test3():
    if dot_product(5, [1, 1, 1, 1, 1], [1, 2, 3, 4, 5]) == 15:
        return True
    return False
    
tester([test1, test2, test3])
