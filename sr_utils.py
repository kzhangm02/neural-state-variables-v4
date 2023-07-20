import sympy
import numpy as np

def rewrite(eq):
    idx2str = {0: ')', 1: ') ** 2', 2: ') ** 3'}
    stack, new_eq = [], ''
    while len(eq) > 0:
        if eq.startswith('square'):
            stack.append(1)
            new_eq += '('
            eq = eq[7:]
        elif eq.startswith('cube'):
            stack.append(2)
            new_eq += '('
            eq = eq[5:]
        elif eq[0] == '(':
            stack.append(0)
            new_eq += '('
            eq = eq[1:]
        elif eq[0] == ')':
            pop = stack[-1]
            stack = stack[:-1]
            new_eq += idx2str[pop]
            eq = eq[1:]
        else:
            new_eq += eq[0]
            eq = eq[1:]
    return new_eq

def simplify(eq, const_threshold):
    final_eq = rewrite(eq)
    eq, final_eq = None, sympy.simplify(final_eq)
    while not eq == final_eq:
        eq = final_eq
        for const in sympy.preorder_traversal(eq):
            if isinstance(const, sympy.Float):
                if np.abs(const) < const_threshold:
                    final_eq = final_eq.subs(const, 0.0)
                elif const < 1.0:
                    final_eq = final_eq.subs(const, float('%.2g' % const))
                else:
                    final_eq = final_eq.subs(const, round(const, 2))
    return str(final_eq)