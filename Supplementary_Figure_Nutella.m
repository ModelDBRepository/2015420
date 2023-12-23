clear all
close all
clc

%% Simulation Parameters/Context Changes 
t1 = 500; %US start 
t2 = 900; %nutella start 
nutellaINT = 30; %nutella duration 
t3 = t2 + nutellaINT*13; %nutella and US end 
t4 = t3 + t1; %start a second US 
T = t4 + 400; %total time

%%  Neuronal Parameters. 
dt = 0.001; %integration time step 
nt = round(T/dt); 
N = 100; %number of neurons 
tm = 0.01; %membrane time constant 
td = 0.02; %decay time in seconds 
tr = 0.002; %rise time in seconds 
tref = 0.002; %refractory time period 
tcar = 0.1; %cacium rise time 
tcad = 1.0; %calcium decay time. 
vreset = -65; %voltage reset 
vpeak = -40; %voltage peak. 
BIAS = vpeak-1; %bias current 
v = vreset + (vpeak-vreset)*rand(N,1);
tspike = zeros(nt,2);
IPSC = zeros(N,1); %post synaptic current storage variable 
h = zeros(N,1); %Storage variable for filtered firing rates
r = zeros(N,1); %second storage variable for filtered rates 
hr = zeros(N,1); %Third variable for filtered rates 
hslow = zeros(N,1); %Storage variable for filtered firing rates
rslow = zeros(N,1); %second storage variable for filtered rates 
hrslow = zeros(N,1); %Third variable for filtered rates 
storev = zeros(nt,10);
storer = zeros(nt,10);
JD = 0*IPSC; %storage variable required for each spike time 
tlast = zeros(N,1); %storage variable for refractory spike times 
ns = 0;% total number of spikes fires 
REC = zeros(T/0.1,N); %store calcium signal. 
sigma = 0.5; %noise variance.  
w = zeros(N,1);
d = 20; %adaptation increment 
tw = 0.1;   %adaptation time constant. 
kd = 0;  %storage initialization 
%% input parameters  
Nin = 1000; %number of inputs. 
omeganutella= -0.1*rand(N,1)+0.01; %Nutella input weight.  
omega = randn(N,Nin)/sqrt(Nin); 
omega0 = omega;
qin = zeros(Nin,1);
nu = 1+zeros(Nin,1);


%% Nutella Simulus 
nutella = zeros(nt,1); 
time = (1:nt)*dt;
for j = 1:1
nutella = nutella + exp(-0.0001*((1:nt)*dt-(t2+j*nutellaINT)).^2)';
end



%% plasticity parameters 
LB = -1;
epsilon = 2e-6; 
plast = 1; 
tau_aux = 400; 
aux = zeros(N,Nin);
iax = randperm(N*Nin,10); 
store_aux = zeros(nt,10);

%% run integration
for i = 1:nt 


%% create Poisson/stochastic input spikes. 
if (dt*i>t1)&(dt*i<t3)|(dt*i>t4)
    nu(1:Nin/2) = 8; else nu(1:Nin/2) = 1;
end
q = rand(Nin,1); 
qin = qin  + (q<nu*dt);
qin = qin*exp(-dt/td); %filter spikes. 
    
%% Integrate neurons and store spikes. 
IPSC = omega*qin;
Inutella = omeganutella*nutella(i);
I = BIAS + IPSC + Inutella - w; %Current to Neurons 
dv = (dt*i>tlast + tref).*(( (-v+I)/tm)); %Voltage equation with refractory period 
v = v + dt*(dv) + sqrt(dt)*sigma*(dt*i>tlast + tref).*randn(N,1)/sqrt(tm); 
w = w + dt*(-w/tw) + d*(v>=vpeak); %apply adaptation variable. 

index = find(v>=vpeak);  %Find the neurons that have spiked 
if length(index)>0
tspike(ns+1:ns+length(index),:) = [index,0*index+dt*i]; %store spike times 
ns = ns + length(index);
end

% Filtered version of neuronal voltages 
if tcar == 0  
rslow = rslow*(1-dt/tcar)+ (v>=vpeak)/tcar;
else
rslow = rslow*(1-dt/tcar) + hrslow*dt; 
hrslow = hrslow*(1-dt/tcad) + (v>=vpeak)/(tcar*tcad);
end




%% reset spikes and apply refractory period. 
tlast = tlast + (dt*i -tlast).*(v>=vpeak);  %Used to set the refractory period of LIF neurons 
v = v + (30 - v).*(v>=vpeak); %rest the voltage and apply a cosmetic spike.  
storev(i,:) = v(1:10);
storer(i,:) = rslow(1:10);
v(v>=vpeak) = vpeak;

store_aux(i,:) = aux(iax);

if mod(i,0.1/dt)==1 
kd = kd + 1; 
REC(kd,:) = rslow;
end



%% apply the plasticity rule. 
if plast == 1
if nutella(i)>0.1
    i0 = 1:N;
aux  = aux(i0,:) + dt*(LB - (mean(omega(i0,:)')'))/tau_aux ;
end
end

aux = aux*exp(-dt/tau_aux);
omega = omega + aux*dt*epsilon;



v = v + (vreset - v).*(v>=vpeak); %reset spike time 

if mod(i,1000)==1
i/nt
end

end
%% 

close all
str = datestr(now,30);
save(sprintf('%s.mat',str))
%% Perform analysis on intervals. 
tlong = ((1:1:kd)*T/kd); 
hc1 = find(tlong<t1-5&tlong>5);
hc2 = find(tlong>t3+5&tlong<t4-5); 
us1 = find(tlong<t2-5&tlong>t1+5); 
us2 = find(tlong>t4+5&tlong<T-5); 
nutella1 = find(tlong<t2+200&tlong>t2-200); 


%% Plot Results. 
figure(1)
US_SCORE1 = trapz(REC(round(t1/0.1):round((t2-100)/0.1),:))/(t2-100-t1);
US_SCORE2 = trapz(REC(round(t4/0.1):round((T-100)/0.1),:))/(T-100-t4);
[rho,pval] = corr(US_SCORE1',US_SCORE2')
plot(US_SCORE1,US_SCORE2,'k.','MarkerSize',14), hold on 
plot([0,20],[0,20],'b')
xlim([0,max(US_SCORE1)])
ylim([0,max(US_SCORE2)])
xlabel('US Score')
ylabel('FRD1 Score') 
title('Area Under Curve Score')

figure(2)
subplot(1,4,1) 
imagesc(tlong(us1),1:N,REC(us1,:)',[0,2.4])
xlim([550,750])
title('US')
subplot(1,4,2)
imagesc(tlong(nutella1),1:N,REC(nutella1,:)',[0,2.4])
xlim([tlong(nutella1(1)),tlong(nutella1(end))])
title('nutella')
subplot(1,4,3)
imagesc(tlong(us2),1:N,REC(us2,:)',[0,2.4])
colormap('jet')
xlim([1900,2100])
title('FRD1')
subplot(1,4,4)
plot(tlong-t1,mean(REC')), hold on 
plot(tlong-t4,mean(REC'))
xlim([20,220])
legend('US','FRD1')
title('US Changes')
%% Apply SVD/PCA
[u,s,v] = svds(REC,3); 
%% 
u(:,1) = -u(:,1); 
u(:,2) = -u(:,2);
%%
figure(3) 
tlong = (1:1:kd)*T/kd; 
hc1 = find(tlong<t1-5&tlong>5);
subplot(1,2,1)
us1 = find(tlong<t2-200&tlong>t1+5); 
us2 = find(tlong>t4+5&tlong<T-200-5); 
nutella1 = find(tlong<t2+50&tlong>t2-50); 
dlong = tlong(2)-tlong(1); 
hold on  
plot(u(us1,1),u(us1,2),'r','LineWidth',2) 
xlabel('PC1')
ylabel('PC2')
plot(u(us2,1),u(us2,2),'b','LineWidth',2), hold on 
plot(u(nutella1,1),u(nutella1,2),'m','LineWidth',2) 
legend('Up state','Up state Post','Nutella')
subplot(1,2,2)
plot(u(us1,1),u(us1,3),'r','LineWidth',2), hold on 
xlabel('PC1')
ylabel('PC3')
plot(u(us2,1),u(us2,3),'b','LineWidth',2) 
plot(u(nutella1,1),u(nutella1,3),'m','LineWidth',2) 




