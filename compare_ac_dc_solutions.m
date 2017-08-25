%% Comparison of the solutions for optimization with AC power flow vs. DC power flow approximation
% This code requires Matpower to be installed and on the Matlab path:
% http://www.pserc.cornell.edu/matpower

% We could ask a related question to what we discussed at our meeting at Northwestern:
% Input the same loads into the AC optimization problem and the DC
% optimization problem. Solve both optimization problems. What is the
% biggest difference between the solutions over a range of possible load
% values?
%
% For any given loading, this is an easy enough question to answer (just
% solve both optimization problems and compare their solutions). A more
% challenging question is to determine the worst-case error between the
% solutions to the optimization problem over a range of loads. Solving this
% directly would require us to formulate a bi-level optimization problem
% (an optimization problem whose constraints include the solution to
% another optimization problem). Bi-level optimization problems are
% generally quite challenging.
%
% So rather than solve the bi-level optimization problem to obtain the
% worst-case, we could train a machine learning algorithm to predict the
% error for a range of load values. The following code sets a random
% loading that is close to some nominal values, then solves the minimum
% cost optimization problems using both the AC power flow and the DC power
% flow approximation. We could repeat this many times to get a large data
% set and then train machine learning algorithms on the result. 

% This should also give you a flavor of using Matpower, solving power
% system optimization problems, etc.
for i = 1:2000
    clear
    clc
    define_constants; 

    %% Problem setup
    mpc = case118; % Name of the data file to load.

    mpopts = mpoption; % contains the options file
    mpopts.opf.dc.solver = 'mips';

    nbus = size(mpc.bus,1);
    ncost = size(mpc.gencost,1);

    % Choose a perturbation of the loads around the nominal values. We may want
    % to try other strategies for coming up with other operating conditions.
    pert = .5; % Maximum amount to perturb the load
    mpc.bus(:,PD) = mpc.bus(:,PD).*(2*pert*rand(nbus,1)+1-pert); % Active power loads
    mpc.bus(:,QD) = mpc.bus(:,QD).*(2*pert*rand(nbus,1)+1-pert); % Reactive power loads
    mpc.gencost(:,5) = mpc.gencost(:,5).*(2*pert*rand(ncost,1)+1-pert);

    %% Run the optimization problem with the AC power flow
    ac_res = runopf(mpc,mpopts);

    %% Run the optimization problem with the AC power flow
    dc_res = rundcopf(mpc,mpopts);
    
    if ac_res.success == false | dc_res.success == false
        continue
    end


    %% Some relevant comparisons

    % Power injections 
    Sac = makeSbus(ac_res.baseMVA,ac_res.bus,ac_res.gen);
    Sdc = makeSbus(dc_res.baseMVA,dc_res.bus,dc_res.gen);

    Pac = real(Sac);
    Qac = real(Sac);

    % Voltages
    Vac_mag = ac_res.bus(:,VM);
    Vac_ang = ac_res.bus(:,VM);

    % Voltage magnitudes are assumed to be 1 by definition in the DC power flow approximation
    Vdc_ang = dc_res.bus(:,VM);

    % Difference in optimal cost
    opt_cost_diff = ac_res.f - dc_res.f;

    % Power in loads, AC and DC are same
    p = reshape(ac_res.bus(:,3), [1], []) ;
    q = reshape(ac_res.bus(:,4), [1], []) ;

    c = reshape(mpc.gencost(:,5), [1], []) ;
    
    dc_line_flow = reshape(abs(dc_res.branch(:,14)), [1],[]);
    dc_line_gen = reshape(abs(dc_res.gen(:,2)), [1], []);
    dc_price = reshape(dc_res.bus(1,14), [1], []);
    
    

%     magnitude_ac_cost = sqrt(ac_res.bus(:,14).^2+ac_res.bus(:,15).^2);
%     ac_price = ac_res.bus(:,14);

%     max_c = max(abs(ac_price - dc_price));
    ac_line_gen = reshape(abs(ac_res.gen(:,2)), [1], []);
     
    
%     max_line_gen_diff = max(abs(ac_line_gen - dc_line_gen));
    



    %% Data formatting
    
    data = [p, q, c, dc_line_gen, ac_line_gen];
    dlmwrite('ac_gen_118.csv',data,'delimiter',',','-append');
end


