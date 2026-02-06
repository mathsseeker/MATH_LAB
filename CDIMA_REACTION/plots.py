from cdima_solve import solve_cdima

sol = solve_cdima(a=10, b=4, x0=0, y0=3, tf=20, n=2000)
t = sol.t
x = sol.y[0]
y = sol.y[1]