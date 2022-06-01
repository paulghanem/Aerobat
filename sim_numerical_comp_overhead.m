% Aerobat matrices execution time comparison simulation


% Objectives of the code:
% 
% 1- computation overhead for D, Cdq, G -> CASadi
% 2- computation overhead for Aerodynamic model -> Estimated Model
% 3 ....
% please further compelete this header with useful info ...

clc ,clear all, close all
import casadi.*


% addpath('generated_functions\') % filesep makes this usable on all OS
addpath(['generated_functions',filesep]) 
addpath(['Utility_functions',filesep])  
addpath(['casadi-windows-matlabR2016a-v3.5.5',filesep]) 

%% aerobat parameters
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
p.n_aero_integration = 5;

params =  [p.mb; p.ma; p.mw; p.g; p.Ib ; p.Ia; p.Iw
    p.L1_L ;p.L2_L; p.L3_L; p.L1_R; p.L2_R; p.L3_R  ]; 
        

% define vars
t_exe = []; % save execution time; order: t_D; t_Cdq; t_G;t_D_CAS; t_Cdq_CAS; t_G_CAS;
t_aero=[];% save execution time of quasi steady state aerodynmaic function and neural net respectively 

% fixed q, dq
 q = zeros(14,1);
 dq = zeros(14,1);
states=[q;dq];

%cASadi functions
func_dyn = external('f', 'func_aerobat_dyn.mexw64');
function_M= external('f', 'function_M.mexw64');
function_Cdq= external('f', 'function_Cdq.mexw64');
function_G= external('f', 'function_G.mexw64');
func_aero_L = external('f', 'func_aerobat_aero_L.mexw64');
func_aero_R = external('f', 'func_aerobat_aero_R.mexw64');

%% neural  net parameters setup
 nh=2;%select number of neurons of the neural net
 Z=zeros(nh,1);% set recurrent weights

 
 M=nh;% number of hidden layers

 paramsNN.M=M;
 nx=size(states,1);%size of NN input
 ny=nx/2;%size of NN output
 paramsNN.nx=nx;
 paramsNN.ny=ny;

% Initialize Neural net parameters
 paramsNN.A = 1e-1*randn(M,nx);
 paramsNN.O=  1e-1*randn(M,nh);
 paramsNN.b = 1e-1*randn(M,1);
 paramsNN.C = 1e-1*randn(ny,M);
 paramsNN.d = zeros(ny,1); % initialize to mean of y


%% compute exec time

% exec time Symbolic funcs

tic
func_M(states,params);
toc
t_exe = [t_exe; toc];

tic
func_Cdq(states,params);
toc
t_exe = [t_exe; toc];

tic
func_G(states,params);
toc
t_exe = [t_exe; toc];

% exec time CasADi funcs

tic
function_M(q,dq,params);
toc
t_exe = [t_exe; toc];

tic
function_Cdq(q,dq,params);
toc
t_exe = [t_exe; toc];

tic
function_G(q,dq,params);
toc
t_exe = [t_exe; toc];


    
% Actuatl aero model exec. time is computed
tic
aero_integration(states,p,func_aero_L,func_aero_R);
toc
t_aero = [t_aero; toc];
% Can you compute the exec. time of your NN model for aero model?
tic
mlpModel(states,Z,paramsNN);%calculate neural net output
toc
t_aero = [t_aero; toc];


%% Matrices execution time plots
figure(1)
%% plots
y1 = t_exe(1:3); % symbolic exec time
y2 = t_exe(4:end); % CasADi exec time

x = categorical({'Inertial matrix','Coriolis matrix','Gravity vector'});
bar(x,y1, 'red'); hold on;
bar(x,y2, 'yellow');
% xlabel('Matrix Type');
ylabel('Execution Time (sec)');
title('Execution Time Comparison. CasADi vs. Matlab Symbolic Toolbox')
legend('Matlab','CasADi')
print(figure(1), '-painters', '-dsvg', '-r600', 'figures\overheadCompDynamics.svg');

%% aerodynamic functions execution time plots
figure(2)
X = categorical({'Actual','Estimated'});
Y=[t_aero];
bar(X,Y,'b');
xlabel('Aerodynamic Models');
ylabel('Execution Time (sec)');
title('Execution Time of Actual and Estimated Models')
print(figure(2), '-painters', '-dsvg', '-r600', 'figures\overheadCompAero.svg');

%% quasi steady state aerodynamic fucntions
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

%% neural network functions
function [H,Z] = mlpModel(X,Z,params)
N = size(X,2);% number of samples
nY = length(params.d);                  % number of outputs
U = params.A*X + 0*params.O*Z+ repmat(params.b,1,N);  % u = Ax + b, x \in R^nX, b,u \in R^nPerceptrons, A \in R^{nP-by-nX}
Z = activationFunction(U);              % z \in R^nP, using nP instead of nPerceptons
V = params.C*Z + repmat(params.d,1,N);  % v = Cz + d, d,v \in R^nY, C \in R^{nY-by-nP}
H = V; % linear output layer activations
%H = exp(V)./repmat(sum(exp(V),1),nY,1); % softmax nonlinearity for second/last layer
% Activate the softmax function to make this MLP a model for class posteriors
%
end 

function out = activationFunction(in)
% Pick a shared nonlinearity for all perceptrons: sigmoid or ramp style...
% You can mix and match nonlinearities in the model.
% However, typically this is not done; identical nonlinearity functions
% are better suited for parallelization of the implementation.
%out = 1./(1+exp(-in)); % Logistic function - sigmoid style nonlinearity
%out = in./sqrt(1+in.^2); % ISRU - ramp style nonlinearity
out=log(1+exp(in));
%out=tanh(in);
%out=max(0,in);

end 












