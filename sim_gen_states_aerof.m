%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Aerobat flapping wing numerical simulation
%
% 4-4-2021
% Author: Paul Ghanem
% Northeastern University, Boston, USA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Simulation Setup

close all, clear all, clc
import casadi.*


% addpath('generated_functions\') % filesep makes this usable on all OS
addpath(['generated_functions',filesep]) 
addpath(['utility_functions',filesep])    
addpath(['simulation_data',filesep])   
addpath(['casadi-windows-matlabR2016a-v3.5.5',filesep]) 


% Simulation parameters
p.t_end = 2;
p.dt = 1e-4;
p.n_aero_integration = 5;


%% Aerobat Parameters

p.g = 9.8; 

% length parameters (about zero configuration)
p.body_radius = 0.025;    % 2.5 cm
p.body_length = 0.1;      % 10cm
p.humerus_length = 0.05;  % 5cm
p.humerus_radius = 0.002;  % 2 mm
p.radius_length = 0.12;   % 13 cm
p.chord_length = 0.1;    % 16 cm
p.wing_thickness = 0.00025;    % 0.25 mm

p.L1_L = [0;0.0207;0.0075];   % Body CoM to shoulder
p.L2_L = [0;p.humerus_length;0];          % shoulder to elbow
p.L3_L = [0;p.radius_length;0];           % elbow to wingtip (midpoint)
p.L1_R = p.L1_L;           
p.L2_R = p.L2_L; 
p.L3_R = p.L3_L;

% mass parameters
p.mb = 0.010; 
p.ma = 0.001; 
p.mw = 0.004; 

% inertia parameters (diagonal components, x-y-z)
p.Ib_1 = p.mb/2*p.body_radius^2;
p.Ib_2 = p.mb/12*(3*p.body_radius^2 + p.body_length^2);
p.Ib_3 = p.mb/12*(3*p.body_radius^2 + p.body_length^2);
p.Ib = [p.Ib_1; p.Ib_2; p.Ib_3]; % cylinder

p.Ia_1 = p.ma/12*(3*p.humerus_radius^2 + p.humerus_length^2);
p.Ia_2 = p.ma/12*(3*p.humerus_radius^2 + p.humerus_length^2);
p.Ia_3 = p.ma/2*p.humerus_radius^2 ; % This inertia is too small
p.Ia = [p.Ia_1; p.Ia_2; p.Ia_3]; % cylinder

p.Iw_1 = p.mw/12*(p.radius_length^2 + p.wing_thickness^2);
p.Iw_2 = p.mw/12*(p.radius_length^2 + p.chord_length^2);
p.Iw_3 = p.mw/12*(p.chord_length^2 + p.wing_thickness^2); % This inertia is kind of small
p.Iw = [p.Iw_1; p.Iw_2; p.Iw_3]; % box

% damping parameters
p.damping_body = [0.01;0.05;0.01];
p.damping_joint = 0.0000;
% p.damping_inertial= 0;

% aerodynamics parameters
p.air_vel = [0;0;0];      % m/s
p.air_density = 1.225;        % kg/m^3

params =  [p.mb; p.ma; p.mw; p.g; p.L1_L ;p.L2_L; p.L3_L; p.L1_R; p.L2_R; p.L3_R;...
            p.Ib ; p.Ia; p.Iw]; 
        


%% Simulation Setup

% Simulation states
t = 0:p.dt:p.t_end;   % time vector
N_sim = length(t); % simulation length

u = zeros(14,1);       % joint torques
ua = zeros(14,1);     % aerodynamics generalized forces

f=15; % flapping frequency

% States order:
% x(1:3) = body inertial position
% x(4:6) = euler angles
% x(7:10) = left wing angles [plunge, medio, elbow, feather]
% x(11:14) = right wing angles [plunge, medio, elbow, feather]
% x(15:17) = body inertial velocity
% x(18:20) = body angular velocity (body frame)
% x(21:24) = left wing angular velocity
% x(25:28) = right wing angular velocity

x = zeros(28,1);% simulation  initale states

%initalizing states at t=0
% x(1)=0;
% x(5)=0;
 x(6)= 360*pi/180;
 x(15)=2;
% x(21)=2*pi*f;
% x(25)=x(21);

%import casadi functions 
func_dyn = external('f', 'func_aerobat_dyn.mexw64');
func_aero_L = external('f', 'func_aerobat_aero_L.mexw64');
func_aero_R = external('f', 'func_aerobat_aero_R.mexw64');

% Data struct to store simulation
data.t = t;
data.x = zeros(length(x), N_sim);
data.u = zeros(length(u),N_sim);
data.ua = zeros(length(ua),N_sim);
data.Cdq = zeros(14,N_sim);
data.angles = zeros(3,N_sim);

data.fw = zeros(6,N_sim);
data.AOA = zeros(2,N_sim);

%body to world rotation matrix
R_body=rotz(x(6))*roty(x(5))*rotx(x(4)); 

%% Numerical Simulation 

tic
for i = 1:N_sim
 
 %body to world rotation matrix
 R_body=rotz(x(6))*roty(x(5))*rotx(x(4));
  
 % Body states positions and velocities
 pos_body = x(1:3);
 vel_body = x(15:17);
  
 %roll pitch yaw
 roll=x(4);
 pitch=x(5);
 yaw=x(6);
 
 %set up desired trajectories for each joint
 t=(i-1)*p.dt;
 wingL_d=sin(2*pi*f*t);
 wingLd_d=2*pi*f*cos(2*pi*f*t);
 wingLdd_d=-(2*pi*f)^2*sin(2*pi*f*t);
 wingR_d=wingL_d;
 wingRd_d=wingLd_d;
 wingRdd_d=wingLdd_d;

 x(7)=wingL_d;
 x(8:10)=zeros(3,1);
 x(11)=wingR_d;
 x(12:14)=zeros(3,1);
 x(21)=wingLd_d;
 x(22:24)=zeros(3,1);
 x(25)=wingRd_d;
 x(26:28)=zeros(3,1);
 
    
  
% get Aerodynamic forces
 [ua,fw,AOA] = aero_integration(x,p,func_aero_L,func_aero_R);
  ua =full(ua);
 
 % get inertia, coriolis and gravity matrices
  [M,Cdq,G] = func_dyn(x(1:14),x(15:28),params) ;
   M=full(M);
   h=Cdq+G;
   
%  x_d=[0;0;0;0;0;0;wingL_d;zeros(3,1);wingR_d;zeros(3,1)];
%  xd_d=[zeros(6,1);wingLd_d;zeros(3,1);wingRd_d;zeros(3,1);];
%  xdd_d=[zeros(6,1);wingLdd_d;zeros(3,1);wingRdd_d;zeros(3,1)];
%u=h -ua+ M*(xdd_d+ Kp.*(-x(1:14)+x_d) +Kd.*(-x(15:28)+xd_d));

%u=ones(14,1)*(1*sin(t)*sin(t/10))^2;
 u=full(u);
  
 
  
  
  % record simulation data ------------------------------------------------
  
  data.x(:,i) = x;
  data.u(:,i) = u;
  data.ua(:,i) = ua;
  data.angles(:,i) = [roll;pitch;yaw];
  data.fw(:,i) = fw;
  data.AOA(:,i) = AOA;
  data.Cdq(:,i) = full(Cdq);
  
  % time integration ------------------------------------------------------
  
  xk = march_rk4(x,u,ua,p,func_dyn);
  x = xk;
  
end
toc

%% Data Plotting
close all
%left and right joints plots 
figure(1)
 subplot(2,2,1), plot(data.t, data.x(7:10,:)/pi*180)
 title('Left joint angles ')
 xlabel('time')
 ylabel('angle in deg')
 subplot(2,2,2), plot(data.t, data.x(11:14,:)/pi*180)
 title('joint angles Right')
 xlabel('time')
 ylabel('angle in deg')
 subplot(2,2,3), plot(data.t, data.x(21:24,:)/pi*180)
 title('joint velocity Left')
 xlabel('time')
 ylabel('angular velocity in deg/s')
 subplot(2,2,4), plot(data.t, data.x(25:28,:)/pi*180)
 title('joint velocity Right')
 xlabel('time')
 ylabel('angular velocity in deg/s')
 % xyz plots
figure(2)
 subplot(2,1,1), plot(data.t, data.x(1:3,:))
 title('body inertial position in m')
 legend('x','y','z')
 subplot(2,1,2), plot(data.t, data.x(15:17,:))
 legend('xdot','ydot','zdot')
 title('body inertial velocity in m/s')
 %roll pitch yaw plots
figure(3)
 subplot(2,1,1),plot(data.t, data.x(4:6,:)/pi*180)
 title('body orientation')
 legend('roll','pitch','yaw')
 xlabel('time')
 ylabel('angle in deg')
 subplot(2,1,2),plot(data.t, data.x(18:20,:)/pi*180)
 title('body Angular velocity')
 legend('wx','wy','wz')
 xlabel('time')
 ylabel('angular velocity in deg/s')
 %aerodynamic forces plots 
 figure(4)
 subplot(2,2,1),plot(data.t, data.fw(1:3,:))
 title('forces on the left wing')
 legend('fx','fy','fz')
 xlabel('time')
 ylabel('force in N')
 subplot(2,2,2),plot(data.t, data.fw(4:6,:))
 title('forces on the right wing')
 legend('fx','fy','fz')
 xlabel('time')
 ylabel('force in N')
 %mean aerodynamic forces plots 
 subplot(2,2,3),plot(data.t, mean(data.fw(1:3,:),2)*ones(1,length(data.t)))
 title('mean forces on the right wing')
 legend('fx','fy','fz')
 xlabel('time')
 ylabel('force in N')
 
 subplot(2,2,4),plot(data.t, mean(data.fw(4:6,:),2)*ones(1,length(data.t)))
 title('mean forces on the right wing')
 legend('fx','fy','fz')
 xlabel('time')
 ylabel('force in N')
 %angle of attach plots 
 figure(5)
 plot(data.t, data.AOA)
 title('angles of attack')
 legend('alpha_L','alpha_R')
 xlabel('time')
 ylabel('angles in rad')

 
 
%% Local Functions

% March RK4 simulation
% x = states, u = inputs, ua = aerodynamic forces, p = parameters
function xk = march_rk4(x,u,ua,p,func_dyn)
f1 = aerobat_model_test(x, u, ua, p,func_dyn);
f2 = aerobat_model_test(x + f1*p.dt/2, u, ua, p,func_dyn);
f3 = aerobat_model_test(x + f2*p.dt/2, u, ua, p,func_dyn);
f4 = aerobat_model_test(x + f3*p.dt, u, ua, p,func_dyn);
xk = x + (f1/6 + f2/3 + f3/3 + f4/6)*p.dt;
end


function [ua,fw,AOA] = aero_integration(x,p,func_aero_L,func_aero_R)

%body to worls rotation
R_body =rot_z(x(6))*rot_y(x(5))*rot_x(x(4)); 


% Armwing rotation matrices
%Reflection_Y=[1 0 0; 0 -1 0;0 0 1];
Rot1_L = rot_z(-x(8))*rot_x(x(7)); 
Rot2_L = rot_x(x(9))*rot_y(-x(10)); 
Rot1_R = rot_z(x(12))*rot_x(-x(11));
Rot2_R = rot_x(-x(13))*rot_y(-x(14));

N_states = 14;            % Dynamic state acceleration length
ua = zeros(N_states,1);   % Generalized forces of the aerodynamic forces
params =  [p.mb; p.ma; p.mw; p.g; p.L1_L ;p.L2_L; p.L3_L; p.L1_R; p.L2_R; p.L3_R;...
            p.Ib ; p.Ia; p.Iw]; 

% setup integration
ds = 1/p.n_aero_integration;

area_ds = ds*p.radius_length*p.chord_length; % area of the segment (assume constant)

% Aerodynamic forces (linear)
f_aero_sum_L = zeros(3,1);
f_aero_sum_R = zeros(3,1);

% integration
for s = ds:ds:1
  
  % Chordwise integration -------------------------------------------------
  
  % distance from elbow to a position at the wing
  Lw_L = [0; p.radius_length;0] * (s - 0.5*ds);    
  Lw_R = [0; p.radius_length;0] * (s - 0.5*ds);
  
  % evaluate leading edge from the center velocity (inertial frame)
  [~,vL] = func_aero_L(x(1:14),x(15:28), params, Lw_L);
  [~,vR] = func_aero_R(x(1:14),x(15:28), params, Lw_R);
  vL=full(vL);
  vR=full(vR);
  
  % Transform to wing frame
  vL_wing = (R_body*Rot1_L*Rot2_L)'*(vL- p.air_vel);
  vR_wing = (R_body*Rot1_R*Rot2_R)'*(vR - p.air_vel);
  
  % Check the leading edge direction and evaluate the aerodynamic forces
  
  % left side
  
  alpha_L = atan2(vL_wing(3), abs(vL_wing(1))); % angle of attack
  if (vL_wing(1) >= 0)
    % Forward facing leading edge
    La_L = [p.chord_length/4; p.radius_length * (s - 0.5*ds); 0];
  else
    % Rear facing leading edge
    La_L = [-p.chord_length/4; p.radius_length * (s - 0.5*ds); 0];
  end
  
  % right side
  alpha_R = atan2(vR_wing(3), abs(vR_wing(1)))  ;       % angle of attack
  if (vR_wing(1) >= 0)
    % Forward facing leading edge
    La_R = [p.chord_length/4; p.radius_length * (s - 0.5*ds); 0];
  else
    % Rear facing leading edge
    La_R = [-p.chord_length/4; p.radius_length * (s - 0.5*ds); 0];
  end
  
  % lift and drag coefficients
  C_lift_L = 5*lift_coefficient_model(alpha_L);
  C_lift_R = 5*lift_coefficient_model(alpha_R);
  C_drag_L = 5*drag_coefficient_model(alpha_L);
  C_drag_R = 5*drag_coefficient_model(alpha_R);
  
  % B matrix for the generalized aerodynamic forces
  [Ba_L, ~] = func_aero_L(x(1:14),x(15:28), params, La_L);
  [Ba_R, ~] = func_aero_R(x(1:14),x(15:28), params, La_R);
  
  % lift and drag forces
  f_lift_L = area_ds*norm(vL_wing([1,3]))^2*C_lift_L*p.air_density/2;
  f_lift_R = area_ds*norm(vR_wing([1,3]))^2*C_lift_R*p.air_density/2;
  f_drag_L = area_ds*norm(vL_wing([1,3]))^2*C_drag_L*p.air_density/2;
  f_drag_R = area_ds*norm(vR_wing([1,3]))^2*C_drag_R*p.air_density/2;
  
  
  % Total aerodynamic forces (inertial frame)
  F_zL=f_lift_L*cos(alpha_L)-f_drag_L*sin(alpha_L);
  F_xL=-f_lift_L*sin(alpha_L)-f_drag_L*cos(alpha_L);
  F_zR=f_lift_R*cos(alpha_R)-f_drag_R*sin(alpha_R);
  F_xR=-f_lift_R*sin(alpha_R)-f_drag_R*cos(alpha_R);

  f_aero_L = (R_body*Rot1_L*Rot2_L)*[F_xL; 0; F_zL];
  f_aero_R = (R_body*Rot1_R*Rot2_R)*[F_xR; 0; F_zR];
  f_aero_sum_L = f_aero_sum_L + f_aero_L;
  f_aero_sum_R = f_aero_sum_R + f_aero_R;
  
  % Generalized forces (use inertial aerodynamic forces)
  ua = ua + Ba_L*f_aero_L; 
  ua = ua + Ba_R*f_aero_R; 

  
  
end
AOA=[alpha_L;alpha_R];
fw = [f_aero_sum_L;f_aero_sum_R];

end

function C_lift = lift_coefficient_model(alpha)
  temp = (2.13 * alpha/pi*180 - 7.2)/180*pi;
  C_lift = 0.225 + 1.58 * sin(temp);
end

function C_drag = drag_coefficient_model(alpha)
  temp = (2.04 * alpha/pi*180 - 9.82)/180*pi;
  C_drag = 1.92 - 1.55 * cos(temp);
end


