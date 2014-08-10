addpath(genpath('~/code/gravlens/'))

data = load('test_nfw_data_gravlenspy.txt');
theta_x = data(:,3);
theta_y = data(:,4);
M_200=4.7779e+14 
conc=4.28229095326
theta_cx=15 
theta_cy=0
zclust= 0.257707923651
zsource=1
Omega_m=0.271
[gamma_1 gamma_2 kappa mod_gamma rho_s r_s] = nfwshears(theta_x,theta_y,M_200,conc,theta_cx,theta_cy,zclust,zsource,Omega_m);

g1_py = data(:,1);
g2_py = data(:,2);

theta_x__1=theta_x(1)
theta_y__1=theta_y(1)

subplot(3,2,1);
scatter(theta_x,theta_y,70,gamma_1,'filled');
title('gravlens matlab')
colorbar;
subplot(3,2,2);
scatter(theta_x,theta_y,70,gamma_2,'filled');
title('gravlens matlab')
colorbar;
subplot(3,2,3);
scatter(theta_x,theta_y,70,g1_py,'filled');
title('python gravlens')
colorbar;
subplot(3,2,4);
scatter(theta_x,theta_y,70,g2_py,'filled');
title('python gravlens')
colorbar;

subplot(3,2,5);
scatter(theta_x,theta_y,70,g1_py./gamma_1,'filled');
title('python / matlab ')
colorbar;

subplot(3,2,6);
scatter(theta_x,theta_y,70,g2_py./gamma_2,'filled');
title('python / matlab')
colorbar;

disp 'now check the plot if its the same'