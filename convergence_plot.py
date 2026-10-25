import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, json
res = json.load(open('C:\\Pratham\\sofa-problem\\results.json'))
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
# LEFT: intersection convergence (O(1/n))
n_vals = [900, 1800, 3600]
area_vals = [res['area_intersection_900'], res['area_intersection_1800'], res['area_intersection_3600']]
ref_val = res['reference']
errors_int = [abs(a - ref_val) for a in area_vals]
axs[0].loglog(1/np.array(n_vals), errors_int, 'o-', color='#c0392b', label='measured error', ms=8, lw=2)
# Theoretical O(1/n) reference line through (1/900, error_900)
axs[0].loglog([1/3600, 1/900], [errors_int[2]*(3600/1800), errors_int[0]*(900/900)], '--', color='#7f8c8d', label='O(1/n) slope')
axs[0].set_xlabel('1 / n')
axs[0].set_ylabel('|area_intersection - reference|')
axs[0].set_title('Intersection method convergence  (O(1/n) measured)')
axs[0].legend(fontsize=8)
axs[0].grid(True, which='both', ls=':')
# Annotate Richardson values
axs[0].annotate('Richardson\n(1800,3600)\n-5.5e-08', xy=(1/1800, errors_int[2]), xytext=(1/2500, errors_int[2]*3), fontsize=7, ha='center', color='#16a085')
# RIGHT: polygon chord error (O(1/n^2))
n_poly = [20000, 40000]
poly_areas = [res['area_boundary_20000'], res['area_boundary_40000']]
poly_errors = [abs(a - ref_val) for a in poly_areas]
axs[1].loglog(1/np.array(n_poly)**2, poly_errors, 's-', color='#8e44ad', ms=8, lw=2, label='measured chord error')
# Reference O(1/n^2) through (1/20000^2, error)
axs[1].loglog([1/40000**2, 1/20000**2], [poly_errors[1]*(40000**2/20000**2), poly_errors[0]*(20000**2/20000**2)], '--', color='#7f8c8d', label='O(1/n^2) slope')
axs[1].set_xlabel('1 / n^2')
axs[1].set_ylabel('|polygon_area - reference|')
axs[1].set_title('Analytic polygon chord error  (O(1/n^2) measured)')
axs[1].legend(fontsize=8)
axs[1].grid(True, which='both', ls=':')
axs[1].annotate('2nd-order\nRichardson\n-1.8e-15', xy=(1/40000**2, poly_errors[1]), xytext=(1/15000**2, poly_errors[1]*4), fontsize=7, ha='center', color='#2980b9')
fig.tight_layout()
fig.savefig('C:\\Pratham\\sofa-problem\\convergence_plot.png', dpi=160)
print('saved convergence_plot.png')
