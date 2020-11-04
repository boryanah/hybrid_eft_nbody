python obtain_fields.py

python obtain_positions.py

mpirun -np 8 python obtain_templates.py

mpirun -np 8 python obtain_power.py

python obtain_covariance_gadget.py

python plot_templates.py

python solve_power.py

python fit_power.py
