%Long range temporal correlations using DFA and modelling broadband EEG
%with ARFIMA for motor intention detection
%Copyright 2018-2021 Maitreyee Wairagkar
%Last Update 18/02/18

%load('Participant1.mat','Participant2.mat');
sig = Participant1; 
Fs=1024;
[b,a]=butter(4,45/(Fs*0.5),'low');                                                         %record time of last sample of the window
chs=[9,10,11];%C3 Cz C4 
wind=windowing_MNW_no_zeros(squeeze(sig{1,1,1}),Fs, 2, 0.1);%get the window i

H = hann(length(downsample(wind{1,1},8)));%hann(length(wind{1,1}));%
%H = gausswin(length(downsample(wind{1,1},8)),1.5);
for p = 1:14
 
    if p ==1, sig = Participant1; num_trials = 40; end 
    if p ==2, sig = Participant2; num_trials = 35; end 
    if p ==3, sig = Participant3; num_trials = 40; end 
    if p ==4, sig = Participant4; num_trials = 40; end 
    if p ==5, sig = Participant5; num_trials = 40; end 
    if p ==6, sig = Participant6; num_trials = 37; end 
    if p ==7, sig = Participant7; num_trials = 40; end 
    if p ==8, sig = Participant8; num_trials = 40; end 
    if p ==9, sig = Participant9; num_trials = 40; end 
    if p ==10,sig = Participant10; num_trials = 40; end 
    if p ==11,sig = Participant11; num_trials = 40; end 
    if p ==12,sig = Participant12; num_trials = 40; end 
    if p ==13,sig = Participant13; num_trials = 40; end 
    if p ==14,sig = Participant14; num_trials = 40; end 
    
for ch=1:3
    %for each trial
    for tr=1:num_trials
        %filter
        tr_rt=filtfilt(b,a,double(squeeze(sig{chs(ch),1,tr})));                %using filtfilt here to maintain phases
        tr_nu=filtfilt(b,a,double(squeeze(sig{chs(ch),2,tr})));
        tr_lt=filtfilt(b,a,double(squeeze(sig{chs(ch),3,tr})));
        
        %create surrogate data by shuffling the phases of full trials
        %instead of windows
        %{
          tr_rt = surrogatedata_shuffle_phase(tr_rt,Fs);
          tr_nu = surrogatedata_shuffle_phase(tr_nu,Fs);
          tr_lt = surrogatedata_shuffle_phase(tr_lt,Fs);
        %}
        %for each window  
       for w=1:length(wind) 
           %find autocorrelation of each window
            %Downsample
            w_rt = downsample(tr_rt(wind{w,1}),8).*H;
            w_nu = downsample(tr_nu(wind{w,1}),8).*H;  
            w_lt = downsample(tr_lt(wind{w,1}),8).*H;
            
            %create surrogate data by shuffling the samples
            %{
            for ii = 1:100
            w_rt = w_rt(randperm(length(w_rt)));
            w_nu = w_nu(randperm(length(w_nu)));
            w_lt = w_lt(randperm(length(w_lt)));
            end
             w_rt = w_rt(randperm(length(w_rt))).*H;
             w_nu = w_nu(randperm(length(w_nu))).*H;
             w_lt = w_lt(randperm(length(w_lt))).*H;
            %}
            
            %create surrogate data by shuffling the phase and keeping the
            %frequency intact as suggested by reviewer 1
            %%{
            w_rt = surrogatedata_shuffle_phase(w_rt,128);%.*H;
            w_nu = surrogatedata_shuffle_phase(w_nu,128);%.*H;
            w_lt = surrogatedata_shuffle_phase(w_lt,128);%.*H;
            %}
          %no downsampling
          %{
            w_rt = tr_rt(wind{w,1}).*H;
            w_nu = tr_nu(wind{w,1}).*H;  
            w_lt = tr_lt(wind{w,1}).*H;
%}
           %Normal DFA - forward backward
           [P(p).DFA_r(ch,tr,w).alpha, P(p).DFA_r(ch,tr,w).n , P(p).DFA_r(ch,tr,w).Fn, P(p).DFA_r(ch,tr,w).intercept,P(p).DFA_r(ch,tr,w).r2,P(p).DFA_r(ch,tr,w).fittedline]=DFA_MNW_forward_backward(w_rt');
           [P(p).DFA_n(ch,tr,w).alpha, P(p).DFA_n(ch,tr,w).n , P(p).DFA_n(ch,tr,w).Fn, P(p).DFA_n(ch,tr,w).intercept,P(p).DFA_n(ch,tr,w).r2,P(p).DFA_n(ch,tr,w).fittedline]=DFA_MNW_forward_backward(w_nu');
           [P(p).DFA_l(ch,tr,w).alpha, P(p).DFA_l(ch,tr,w).n , P(p).DFA_l(ch,tr,w).Fn, P(p).DFA_l(ch,tr,w).intercept,P(p).DFA_l(ch,tr,w).r2,P(p).DFA_l(ch,tr,w).fittedline]=DFA_MNW_forward_backward(w_lt');
           
          %smoothing of DFA exponents using exponential smoothing
          %{
           if(w>1)
               dfa_r = expsmooth([P(p).DFA_r(ch,tr,w-1).alpha;P(p).DFA_r(ch,tr,w).alpha],1/0.1,300);
               P(p).DFA_r(ch,tr,w).alpha = dfa_r(2);
               dfa_n = expsmooth([P(p).DFA_n(ch,tr,w-1).alpha;P(p).DFA_n(ch,tr,w).alpha],1/0.1,300);
               P(p).DFA_n(ch,tr,w).alpha = dfa_n(2);
               dfa_l = expsmooth([P(p).DFA_l(ch,tr,w-1).alpha;P(p).DFA_l(ch,tr,w).alpha],1/0.1,300);
               P(p).DFA_l(ch,tr,w).alpha = dfa_l(2);
           end
          %}
           %{
           %Alternative model comparison ML_DFA using AIC BIC
            [P(p).DFA_r(ch,tr,w).AIC, P(p).DFA_r(ch,tr,w).BIC, P(p).DFA_r(ch,tr,w).A, P(p).DFA_r(ch,tr,w).ft, P(p).DFA_r(ch,tr,w).pdf, P(p).DFA_r(ch,tr,w).LLp, P(p).DFA_r(ch,tr,w).r2all, P(p).DFA_r(ch,tr,w).modelAIC, P(p).DFA_r(ch,tr,w).modelBIC] = Comparison_AIC_BIC(P(p).DFA_r(ch,tr,w).n,P(p).DFA_r(ch,tr,w).Fn);
            [P(p).DFA_n(ch,tr,w).AIC, P(p).DFA_n(ch,tr,w).BIC, P(p).DFA_n(ch,tr,w).A, P(p).DFA_n(ch,tr,w).ft, P(p).DFA_n(ch,tr,w).pdf, P(p).DFA_n(ch,tr,w).LLp, P(p).DFA_n(ch,tr,w).r2all, P(p).DFA_n(ch,tr,w).modelAIC, P(p).DFA_n(ch,tr,w).modelBIC] = Comparison_AIC_BIC(P(p).DFA_n(ch,tr,w).n,P(p).DFA_n(ch,tr,w).Fn);
            [P(p).DFA_l(ch,tr,w).AIC, P(p).DFA_l(ch,tr,w).BIC, P(p).DFA_l(ch,tr,w).A, P(p).DFA_l(ch,tr,w).ft, P(p).DFA_l(ch,tr,w).pdf, P(p).DFA_l(ch,tr,w).LLp, P(p).DFA_l(ch,tr,w).r2all, P(p).DFA_l(ch,tr,w).modelAIC, P(p).DFA_l(ch,tr,w).modelBIC] = Comparison_AIC_BIC(P(p).DFA_l(ch,tr,w).n,P(p).DFA_l(ch,tr,w).Fn);
           %}
           
       end %end w
       disp([p ch tr ])
    end%end tr
end%end ch
end%Participant

%% PLOT GRAND AVERAGE alpha - aggregate the results %%%%%%%%%%%%%%%%
cnt = 1;
for p =1:14 %%%
    for tr = 1:size(P(p).DFA_r,2)
       for ch =1:3 %%%%
            for w = 1:size(P(p).DFA_r,3)
               %%{
                dfaR(ch,cnt,w)=P(p).DFA_r(ch,tr,w).alpha; 
                dfaN(ch,cnt,w)=P(p).DFA_n(ch,tr,w).alpha; 
                dfaL(ch,cnt,w)=P(p).DFA_l(ch,tr,w).alpha; 
                
                r2R(ch,cnt,w)=P(p).DFA_r(ch,tr,w).r2; 
                r2N(ch,cnt,w)=P(p).DFA_n(ch,tr,w).r2; 
                r2L(ch,cnt,w)=P(p).DFA_l(ch,tr,w).r2; 
                %}
                %{
                modAICR(ch,cnt,w)=P(p).DFA_r(ch,tr,w).modelAIC; 
                modAICN(ch,cnt,w)=P(p).DFA_n(ch,tr,w).modelAIC; 
                modAICL(ch,cnt,w)=P(p).DFA_l(ch,tr,w).modelAIC; 
                
                modBICR(ch,cnt,w)=P(p).DFA_r(ch,tr,w).modelBIC; 
                modBICN(ch,cnt,w)=P(p).DFA_n(ch,tr,w).modelBIC; 
                modBICL(ch,cnt,w)=P(p).DFA_l(ch,tr,w).modelBIC; 
                %}
            end
       end
        cnt = cnt+1;
    end
end

%% Look at coefficients of second order plot
cnt = 1;
cnt2 = 1;
dist_2 = [];
dist_1 = [];
for p =1:14
    for tr = 1:size(P(p).DFA_r,2)
       for ch =1:1:size(P(p).DFA_r,1)
            for w = 1:size(P(p).DFA_r,3)
                if(P(p).DFA_r(ch,tr,w).modelAIC == 2)
                   quadcoef(cnt) = P(p).DFA_r(ch,tr,w).A{2}(1);
                   lincoef(cnt) = P(p).DFA_r(ch,tr,w).A{2}(2);
                   cnt = cnt+1;
                   dist_2 = [dist_2,(log2(P(1).DFA_r(1,1,3).Fn)-P(1).DFA_r(1,1,3).ft(2,:))];
                elseif(P(p).DFA_r(ch,tr,w).modelAIC == 1)
                    linearcoef(cnt2) = P(p).DFA_r(ch,tr,w).A{1}(1);
                    dist_1 = [dist_1,(log2(P(1).DFA_r(1,1,3).Fn)-P(1).DFA_r(1,1,3).ft(1,:))];
                    cnt2 = cnt2+1;
                end
                
                if(P(p).DFA_n(ch,tr,w).modelAIC == 2)
                   quadcoef(cnt) = P(p).DFA_n(ch,tr,w).A{2}(1);
                   lincoef(cnt) = P(p).DFA_n(ch,tr,w).A{2}(2);
                   dist_2 = [dist_2,(log2(P(1).DFA_n(1,1,3).Fn)-P(1).DFA_n(1,1,3).ft(2,:))];
                   cnt = cnt+1;
                elseif(P(p).DFA_n(ch,tr,w).modelAIC == 1)
                    linearcoef(cnt2) = P(p).DFA_n(ch,tr,w).A{1}(1);
                    dist_1 = [dist_1,(log2(P(1).DFA_n(1,1,3).Fn)-P(1).DFA_n(1,1,3).ft(1,:))];
                    cnt2 = cnt2+1;
                end
                
                if(P(p).DFA_l(ch,tr,w).modelAIC == 2)
                   quadcoef(cnt) = P(p).DFA_l(ch,tr,w).A{2}(1);
                   lincoef(cnt) = P(p).DFA_l(ch,tr,w).A{2}(2);
                   dist_2 = [dist_2,(log2(P(1).DFA_l(1,1,3).Fn)-P(1).DFA_l(1,1,3).ft(2,:))];
                   cnt = cnt+1;
                elseif(P(p).DFA_l(ch,tr,w).modelAIC == 1)
                    linearcoef(cnt2) = P(p).DFA_l(ch,tr,w).A{1}(1);
                    dist_1 = [dist_1,(log2(P(1).DFA_l(1,1,3).Fn)-P(1).DFA_l(1,1,3).ft(1,:))];
                    cnt2 = cnt2+1;
                end
           
            end
        end
    end
end

figure,
subplot(1,3,1),histogram(quadcoef,'Normalization' ,'probability');title('Distribution of quadratic coefficients');
subplot(1,3,2),histogram(lincoef,'Normalization' ,'probability');title('Distribution of linear coefficients');
subplot(1,3,3),histogram(linearcoef,'Normalization' ,'probability');title('Distribution of linear');

figure,
subplot(1,3,1),histogram(dist_2,'Normalization' ,'probability');title('Distribution of quadratic remainders');
subplot(1,3,2),histogram(dist_1,'Normalization' ,'probability');title('Distribution of linear remainders');
%% Plot histogram of model selected
cnt = 1;
for i = 1:size(modAICR,1)
    for j = 1:size(modAICR,2)
        for k = 1:size(modAICR,3)
            if(modAICR(i,j,k) == 1 || modBICR(i,j,k)== 1)
               aicmod (cnt) = 1;
            else
                aicmod (cnt) = modAICR(i,j,k);
            end
            cnt = cnt+1;
            if(modAICN(i,j,k) == 1 || modBICN(i,j,k)== 1)
               aicmod (cnt) = 1;
            else
                aicmod (cnt) = modAICR(i,j,k);
            end
            cnt = cnt+1;
            if(modAICL(i,j,k) == 1 || modBICL(i,j,k)== 1)
               aicmod (cnt) = 1;
            else
                aicmod (cnt) = modAICR(i,j,k);
            end
            cnt = cnt+1;
            
        end
    end
end

%aicmod = [reshape(modAICR,[1 numel(modAICR)]), reshape(modAICN,[1 numel(modAICR)]),reshape(modAICL,[1 numel(modAICR)])];
%bicmod = [reshape(modBICR,[1 numel(modAICR)]), reshape(modBICN,[1 numel(modAICR)]),reshape(modBICL,[1 numel(modAICR)])];
figure,
subplot(1,2,1),histogram(aicmod,'Normalization' ,'probability');title('Models selected by AIC');
%subplot(1,2,2),histogram(bicmod,'Normalization' ,'probability');title('Models selected by BIC');

%% Check if the contribution of quadratic is small by looking at quadratic term / linear term ratio in cases where quadratic is selected
cnt = 1;
ratio_r =[];
ratio_n =[];
ratio_l =[];
n = P(1).DFA_r(1,1,1).n;
for p =1:14 %
    for tr = 1:size(P(p).DFA_r,2)
       for ch =1:3 
            for w = 1:size(P(p).DFA_r,3)
                
                if(P(p).DFA_r(ch,tr,w).modelAIC == 2) 
                    A= P(p).DFA_r(ch,tr,w).A{2};
                    ratio_r  = [ratio_r ,A(1)/A(2) ];%(A(1)*log2(n))./A(2)
                end
                
                if(P(p).DFA_n(ch,tr,w).modelAIC == 2) 
                    A= P(p).DFA_n(ch,tr,w).A{2};
                    ratio_n  = [ratio_n , A(1)/A(2)];%(A(1)*log2(n))./A(2)
                end
                
                 if(P(p).DFA_l(ch,tr,w).modelAIC == 2) 
                    A= P(p).DFA_l(ch,tr,w).A{2};
                    ratio_l  = [ratio_l , A(1)/A(2)];%(A(1)*log2(n))./A(2)
                end
                
            end 
            disp( [p tr ch])
        end
    end
end

histogram([ratio_r ratio_n ratio_l],'Normalization' ,'probability');title('Ratio of 2nd order coefficient to 1st order coefficient in the cases where quadratic model was selected');

%% R2
rsq = [reshape(r2R,[1 numel(r2R)]), reshape(r2N,[1 numel(r2N)]),reshape(r2L,[1 numel(r2L)])];
h = histogram(rsq);title('R\^{2}');
%%
%%Plot error shard bars for alpha %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PAPER 2 FIGURE 1 - grand average
fs =12;
time = -1:0.1:3;
figure,hold on;
% single trial plotting
for p =1:14 %%%
 for ch =1:3 %%%%
    for tr = 1:size(P(p).DFA_r,2)
       
            for w = 1:size(P(p).DFA_r,3)
                dfa_r(ch,tr,w)=P(p).DFA_r(ch,tr,w).alpha; 
                dfa_n(ch,tr,w)=P(p).DFA_n(ch,tr,w).alpha; 
                dfa_l(ch,tr,w)=P(p).DFA_l(ch,tr,w).alpha; 
            end

    end
       dfaR(ch,p,:) = squeeze(mean(dfa_r(ch,:,:)));
       dfaN(ch,p,:) = squeeze(mean(dfa_n(ch,:,:)));
       dfaL(ch,p,:) = squeeze(mean(dfa_l(ch,:,:)));
 end

end

pp=[1:7,1:7];
col=[1,7,13,19,25,31,37,4,10,16,22,28,34,40];

for p = 1:14

subplot(7,6,col(p)),plot(time,squeeze(dfaR(1,p,:)),'r','Linewidth',2); hold on;
               plot(time,squeeze(dfaN(1,p,:)),'k','Linewidth',2); 
               plot(time,squeeze(dfaL(1,p,:)),'b','Linewidth',2); xlim([time(1) time(end)]);

subplot(7,6,col(p)+1),plot(time,squeeze(dfaR(2,p,:)),'r','Linewidth',2); hold on;
               plot(time,squeeze(dfaN(2,p,:)),'k','Linewidth',2); 
               plot(time,squeeze(dfaL(2,p,:)),'b','Linewidth',2); xlim([time(1) time(end)]);

subplot(7,6,col(p)+2),plot(time,squeeze(dfaR(3,p,:)),'r','Linewidth',2); hold on;
               plot(time,squeeze(dfaN(3,p,:)),'k','Linewidth',2); 
               plot(time,squeeze(dfaL(3,p,:)),'b','Linewidth',2); xlim([time(1) time(end)]);

               

end %end p
%% grand average%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fs =12;
time = -1:0.1:3;
for p =1:14 %%%
 for ch =1:3
    for tr = 1:size(P(p).DFA_r,2)
       
            for w = 1:size(P(p).DFA_r,3)
                dfa_r(ch,tr,w)=P(p).DFA_r(ch,tr,w).alpha; 
                dfa_n(ch,tr,w)=P(p).DFA_n(ch,tr,w).alpha; 
                dfa_l(ch,tr,w)=P(p).DFA_l(ch,tr,w).alpha; 
            end

    end
       dfaR(ch,p,:) = squeeze(mean(dfa_r(ch,:,:)));
       dfaN(ch,p,:) = squeeze(mean(dfa_n(ch,:,:)));
       dfaL(ch,p,:) = squeeze(mean(dfa_l(ch,:,:)));
 end
end

%normalise according to the mean
for tr = 1:size(dfaR,2)
for ch  = 1:size(dfaR,1)
    br = mean(dfaR(ch,tr,:)-mean(dfaR(ch,:,:)));
    bn = mean(dfaN(ch,tr,:)-mean(dfaN(ch,:,:)));
    bl = mean(dfaL(ch,tr,:)-mean(dfaL(ch,:,:)));
    dfaR(ch,tr,:)=dfaR(ch,tr,:)-br;
    dfaN(ch,tr,:)=dfaN(ch,tr,:)-bn;
    dfaL(ch,tr,:)=dfaL(ch,tr,:)-bl;
end
end

figure,
subplot(1,3,1),shadedErrorBar(time,mean(squeeze(dfaR(1,:,:))),[prctile(squeeze(dfaR(1,:,:)),75);prctile(squeeze(dfaR(1,:,:)),25)],{'r','Linewidth',2},1); hold on;%std(squeeze(dfaR(1,:,:)))
               shadedErrorBar(time,mean(squeeze(dfaN(1,:,:))),[prctile(squeeze(dfaN(1,:,:)),75);prctile(squeeze(dfaN(1,:,:)),25)],{'k','Linewidth',2},1);
               shadedErrorBar(time,mean(squeeze(dfaL(1,:,:))),[prctile(squeeze(dfaL(1,:,:)),75);prctile(squeeze(dfaL(1,:,:)),25)],{'b','Linewidth',2},1); 
xlabel('Time (s)'); ylabel('DFA scaling exponent');title('C3');xlim([time(1) time(end)]); set(gca,'fontsize',fs);

subplot(1,3,2),shadedErrorBar(time,mean(squeeze(dfaR(2,:,:))),[prctile(squeeze(dfaR(2,:,:)),75);prctile(squeeze(dfaR(2,:,:)),25)],{'r','Linewidth',2},1); hold on;%std(squeeze(dfaR(1,:,:)))
               shadedErrorBar(time,mean(squeeze(dfaN(2,:,:))),[prctile(squeeze(dfaN(2,:,:)),75);prctile(squeeze(dfaN(2,:,:)),25)],{'k','Linewidth',2},1);
               shadedErrorBar(time,mean(squeeze(dfaL(2,:,:))),[prctile(squeeze(dfaL(2,:,:)),75);prctile(squeeze(dfaL(2,:,:)),25)],{'b','Linewidth',2},1); 
xlabel('Time (s)'); ylabel('DFA scaling exponent');title('Cz');xlim([time(1) time(end)]); set(gca,'fontsize',fs);

subplot(1,3,3),shadedErrorBar(time,mean(squeeze(dfaR(3,:,:))),[prctile(squeeze(dfaR(3,:,:)),75);prctile(squeeze(dfaR(3,:,:)),25)],{'r','Linewidth',2},1); hold on;%std(squeeze(dfaR(1,:,:)))
               shadedErrorBar(time,mean(squeeze(dfaN(3,:,:))),[prctile(squeeze(dfaN(3,:,:)),75);prctile(squeeze(dfaN(3,:,:)),25)],{'k','Linewidth',2},1);
               shadedErrorBar(time,mean(squeeze(dfaL(3,:,:))),[prctile(squeeze(dfaL(3,:,:)),75);prctile(squeeze(dfaL(3,:,:)),25)],{'b','Linewidth',2},1); 
xlabel('Time (s)'); ylabel('DFA scaling exponent');title('C4');xlim([time(1) time(end)]); set(gca,'fontsize',fs);
%%
%%plot box plots for r2 %%%%%%%%%%%%%%
%figure, boxplot(squeeze(r2R(1,:,:)),'PlotStyle','compact','Color','r');ylabel('R2'); title('C3 Right');%set(gca,'xticklabel',{' '});
%figure, boxplot(squeeze(r2N(1,:,:)),'PlotStyle','compact','Color','k');ylabel('R2'); title('C3 Neutral');%set(gca,'xticklabel',{' '});
%figure, boxplot(squeeze(r2L(1,:,:)),'PlotStyle','compact','Color','b');ylabel('R2'); title('C3 Left');%set(gca,'xticklabel',{' '});
figure,
subplot(3,3,1), boxplot(squeeze(r2R(1,:,:)),'PlotStyle','compact','Color','r');ylabel('R2'); title('C3 Right');%set(gca,'xticklabel',{' '});
%subplot(3,3,2), boxplot(squeeze(r2R(2,:,:)),'PlotStyle','compact','Color','r');ylabel('R2'); title('Cz Right');%set(gca,'xticklabel',{' '});
%subplot(3,3,3), boxplot(squeeze(r2R(3,:,:)),'PlotStyle','compact','Color','r');ylabel('R2'); title('C4 Right');%set(gca,'xticklabel',{' '});

subplot(3,3,4), boxplot(squeeze(r2N(1,:,:)),'PlotStyle','compact','Color','k');ylabel('R2'); title('C3 Neutral');%set(gca,'xticklabel',{' '});
%subplot(3,3,5), boxplot(squeeze(r2N(2,:,:)),'PlotStyle','compact','Color','k');ylabel('R2'); title('Cz Neutral');%set(gca,'xticklabel',{' '});
%subplot(3,3,6), boxplot(squeeze(r2N(3,:,:)),'PlotStyle','compact','Color','k');ylabel('R2'); title('C4 Neutral');%set(gca,'xticklabel',{' '});

subplot(3,3,7), boxplot(squeeze(r2L(1,:,:)),'PlotStyle','compact','Color','b');ylabel('R2'); title('C3 Left');%set(gca,'xticklabel',{' '});
%subplot(3,3,8), boxplot(squeeze(r2L(2,:,:)),'PlotStyle','compact','Color','b');ylabel('R2'); title('Cz Left');%set(gca,'xticklabel',{' '});
%subplot(3,3,9), boxplot(squeeze(r2L(3,:,:)),'PlotStyle','compact','Color','b');ylabel('R2'); title('C4 Left');%set(gca,'xticklabel',{' '});
%%
%%{
figure,
cnt = 0;
for tr = 1:35
    plot((1:41)+cnt,squeeze(dfaL(1,tr,:)),'b'); hold on;
   % plot(squeeze(dfaL(1,tr,:)),'b'); hold on;
    cnt = cnt +41;
end
figure,
cnt = 0;
for tr = 1:35
    plot((1:41)+cnt,squeeze(dfaR(1,tr,:)),'r'); hold on;
  % plot(squeeze(dfaR(1,tr,:)),'r'); hold on;
    cnt = cnt +41;
end
figure,
cnt = 0;
for tr = 1:35
    plot((1:41)+cnt,squeeze(dfaN(1,tr,:)),'k'); hold on;
   % plot(squeeze(dfaN(1,tr,:)),'k'); hold on;
    cnt = cnt +41;
end
%}
%% T test
for i = 1:41
    [h,p_val_rt(i)] = ttest2(dfaR(1,:,i), dfaN(1,:,i));
    [h,p_val_lt(i)] = ttest2(dfaL(1,:,i),dfaN(1,:,i));
end
figure,
plot(time,p_val_rt,'r','Linewidth',2);hold on;
plot(time,p_val_lt,'b','Linewidth',2);xlabel('time (s)'); ylabel('p value');
plot(time,ones(1,length(time)).*0.05,'k');

%% Detection of long range dependence using ARFIMA -compare short range dependence ARMA(1,1) with ARFIMA(1,d,1)

sig = Participant1; 
Fs=1024;
[b,a]=butter(4,45/(Fs*0.5),'low');                                                       %record time of last sample of the window
chs=[9,10,11];%C3 Cz C4 
wind=windowing_MNW_no_zeros(squeeze(sig{1,1,1}),Fs, 2, 0.1);%get the window i

H = hann(length(downsample(wind{1,1},8)));%hann(length(wind{1,1}));%
for p = 1:14
 
    if p ==1, sig = Participant1; num_trials = 40; end 
    if p ==2, sig = Participant2; num_trials = 35; end 
    if p ==3, sig = Participant3; num_trials = 40; end 
    if p ==4, sig = Participant4; num_trials = 40; end 
    if p ==5, sig = Participant5; num_trials = 40; end 
    if p ==6, sig = Participant6; num_trials = 37; end 
    if p ==7, sig = Participant7; num_trials = 40; end 
    if p ==8, sig = Participant8; num_trials = 40; end 
    if p ==9, sig = Participant9; num_trials = 40; end 
    if p ==10,sig = Participant10; num_trials = 40; end 
    if p ==11,sig = Participant11; num_trials = 40; end 
    if p ==12,sig = Participant12; num_trials = 40; end 
    if p ==13,sig = Participant13; num_trials = 40; end 
    if p ==14,sig = Participant14; num_trials = 40; end 
    
for ch=2:3 
    %for each trial
    for tr=1:num_trials
        %filter
        tr_rt=filtfilt(b,a,double(squeeze(sig{chs(ch),1,tr})));                %using filtfilt here to maintain phases
        tr_nu=filtfilt(b,a,double(squeeze(sig{chs(ch),2,tr})));
        tr_lt=filtfilt(b,a,double(squeeze(sig{chs(ch),3,tr})));
        
        %for each window  
       for w=1:length(wind) 
           %find autocorrelation of each window
            %Downsample %To .*H or not to .*H ?
            w_rt = downsample(tr_rt(wind{w,1}),8);
            w_nu = downsample(tr_nu(wind{w,1}),8);  
            w_lt = downsample(tr_lt(wind{w,1}),8);
           
%%{
%fit ARIMA and ARMA
[PP(p).DFA_r(ch,tr,w).mdl, PP(p).DFA_r(ch,tr,w).mdlARFIMA , PP(p).DFA_r(ch,tr,w).loglARFIMA,...
 PP(p).DFA_r(ch,tr,w).mdlARMA , PP(p).DFA_r(ch,tr,w).loglARMA]=short_long_range_determination(P(p).DFA_r(ch,tr,w).alpha,w_rt);
 
[PP(p).DFA_n(ch,tr,w).mdl, PP(p).DFA_n(ch,tr,w).mdlARFIMA , PP(p).DFA_n(ch,tr,w).loglARFIMA,...
 PP(p).DFA_n(ch,tr,w).mdlARMA , PP(p).DFA_n(ch,tr,w).loglARMA]=short_long_range_determination(P(p).DFA_n(ch,tr,w).alpha,w_nu);

[PP(p).DFA_l(ch,tr,w).mdl, PP(p).DFA_l(ch,tr,w).mdlARFIMA , PP(p).DFA_l(ch,tr,w).loglARFIMA,...
 PP(p).DFA_l(ch,tr,w).mdlARMA , PP(p).DFA_l(ch,tr,w).loglARMA]=short_long_range_determination(P(p).DFA_l(ch,tr,w).alpha,w_lt);
%}

       end %end w
       disp([p ch tr])
    end%end tr
end%end ch
end%Participant
%%
% plot histogram of how many windows, ARFIMA was chosen
cnt = 1;
for p = 1:14
    for ch = 1:1%size(P(p).DFA_r,1)
    for tr = 1:size(PP(p).DFA_r,2)
        for w = 1:size(PP(p).DFA_r,3)
            mdl(cnt)= PP(p).DFA_r(ch,tr,w).mdl;%ARFIMA.P ;
            cnt = cnt+1;
        end
    end
    end
end
 
figure,
histogram(mdl,'Normalization' ,'probability');title('Models selected by AIC');




%% fit ARMA model to the evolution of dynamics of DFA exponent
%M is model
for ch = 1:size(dfaR(:,:,:),1)
    for tr=1:size(dfaR(:,:,:),2)
        
        [M.DFA_r(ch,tr).p,M.DFA_r(ch,tr).mdl,M.DFA_r(ch,tr).LogL,M.DFA_r(ch,tr).stdr, ...
         M.DFA_r(ch,tr).E,M.DFA_r(ch,tr).h,M.DFA_r(ch,tr).t,M.DFA_r(ch,tr).z, ...
         M.DFA_r(ch,tr).YF,M.DFA_r(ch,tr).YMSE]= ARMA_for_DFA_exponents(squeeze(dfaR(ch,tr,:)));
       
        [M.DFA_n(ch,tr).p,M.DFA_n(ch,tr).mdl,M.DFA_n(ch,tr).LogL,M.DFA_n(ch,tr).stdr, ...
         M.DFA_n(ch,tr).E,M.DFA_n(ch,tr).h,M.DFA_n(ch,tr).t,M.DFA_n(ch,tr).z, ...
         M.DFA_n(ch,tr).YF,M.DFA_n(ch,tr).YMSE]= ARMA_for_DFA_exponents(squeeze(dfaN(ch,tr,:)));
       
        [M.DFA_l(ch,tr).p,M.DFA_l(ch,tr).mdl,M.DFA_l(ch,tr).LogL,M.DFA_l(ch,tr).stdr, ...
         M.DFA_l(ch,tr).E,M.DFA_l(ch,tr).h,M.DFA_l(ch,tr).t,M.DFA_l(ch,tr).z, ...
         M.DFA_l(ch,tr).YF,M.DFA_l(ch,tr).YMSE]= ARMA_for_DFA_exponents(squeeze(dfaL(ch,tr,:)));
       
     disp([ch,tr])
    end
end
%%
for ch = 1:3
    cnt2r=1; cnt2n=1;  cnt2l=1;
    cnt1r=1; cnt1n=1;  cnt1l=1;
    for tr=1:size(M.DFA_l,2)
        %Right
        if(M.DFA_r(ch,tr).p == 2 && M.DFA_r(ch,tr).h ==0)
            mdl1r(ch,cnt2r) = M.DFA_r(ch,tr).mdl.AR{1};
            mdl2r(ch,cnt2r) = M.DFA_r(ch,tr).mdl.AR{2};
            Hr_2(ch,cnt2r)= M.DFA_r(ch,tr).h;
            cnt2r = cnt2r+1;
        end 
        if(M.DFA_r(ch,tr).p == 1)
            Hr_1(ch,cnt1r)= M.DFA_r(ch,tr).h;
            cnt1r = cnt1r+1;
        end 
         %Neutral
        if(M.DFA_n(ch,tr).p == 2 &&  M.DFA_n(ch,tr).h ==0)
            mdl1n(ch,cnt2n) = M.DFA_n(ch,tr).mdl.AR{1};
            mdl2n(ch,cnt2n) = M.DFA_n(ch,tr).mdl.AR{2};
            Hn_2(ch,cnt2n)= M.DFA_n(ch,tr).h;
            cnt2n = cnt2n+1;
        end 
        if(M.DFA_n(ch,tr).p == 1)
            Hn_1(ch,cnt1n)= M.DFA_n(ch,tr).h;
            cnt1n = cnt1n+1;
        end 
        %Left
        if(M.DFA_l(ch,tr).p == 2 &&  M.DFA_l(ch,tr).h ==0)
            mdl1l(ch,cnt2l) = M.DFA_l(ch,tr).mdl.AR{1};
            mdl2l(ch,cnt2l) = M.DFA_l(ch,tr).mdl.AR{2};
            Hl_2(ch,cnt2l)= M.DFA_l(ch,tr).h;
            cnt2l = cnt2l+1;
        end 
        if(M.DFA_l(ch,tr).p == 1)
            Hl_1(ch,cnt1l)= M.DFA_l(ch,tr).h;
            cnt1l = cnt1l+1;
        end 
    end
end
%% 
H2 = [reshape(Hr_2,1,numel(Hr_2)),reshape(Hn_2,1,numel(Hn_2)),reshape(Hl_2,1,numel(Hl_2))];
H1 = [reshape(Hr_1,1,numel(Hr_1)),reshape(Hn_1,1,numel(Hn_1)),reshape(Hl_1,1,numel(Hl_1))];
figure,histogram(H2,'Normalization','probability')
figure,histogram(H1,'Normalization','probability')
%% plot the model parameters
%{
subplot(2,3,4),
bins = 20;
h1=histfit(mdl2r(1,:),bins,'kernel');hold on;
h2=histfit(mdl2n(1,:),bins,'kernel');title('C3: AR{2}');
h3=histfit(mdl2l(1,:),bins,'kernel');xlim([-1.1 0]);
h1(1).Visible = 'off';
h2(1).Visible = 'off';
h3(1).Visible = 'off';
h1(2).Color = 'r';
h2(2).Color = 'k';
h3(2).Color = 'b';
clear h1 h2 h3;

subplot(2,3,1),
h11=histfit(mdl1r(1,:),bins,'kernel');hold on;
h22=histfit(mdl1n(1,:),bins,'kernel');xlim([0 2.3]);
h33=histfit(mdl1l(1,:),bins,'kernel');title('C3: AR{1}');
h11(1).Visible = 'off';
h22(1).Visible = 'off';
h33(1).Visible = 'off';
h11(2).Color = 'r';
h22(2).Color = 'k';
h33(2).Color = 'b';
clear h11 h22 h33;

subplot(2,3,5)
h1=histfit(mdl2r(2,:),bins,'kernel');hold on;
h2=histfit(mdl2n(2,:),bins,'kernel');title('Cz: AR{2}');
h3=histfit(mdl2l(2,:),bins,'kernel');xlim([-1.1 0]);
h1(1).Visible = 'off';
h2(1).Visible = 'off';
h3(1).Visible = 'off';
h1(2).Color = 'r';
h2(2).Color = 'k';
h3(2).Color = 'b';
clear h1 h2 h3;

subplot(2,3,2),
h11=histfit(mdl1r(2,:),bins,'kernel');hold on;
h22=histfit(mdl1n(2,:),bins,'kernel');xlim([0 2.3]);
h33=histfit(mdl1l(2,:),bins,'kernel');title('Cz: AR{1}');
h11(1).Visible = 'off';
h22(1).Visible = 'off';
h33(1).Visible = 'off';
h11(2).Color = 'r';
h22(2).Color = 'k';
h33(2).Color = 'b';
clear h11 h22 h33;

subplot(2,3,6)
h1=histfit(mdl2r(3,:),bins,'kernel');hold on;
h2=histfit(mdl2n(3,:),bins,'kernel');title('C4: AR{2}');
h3=histfit(mdl2l(3,:),bins,'kernel');xlim([-1.1 0]);
h1(1).Visible = 'off';
h2(1).Visible = 'off';
h3(1).Visible = 'off';
h1(2).Color = 'r';
h2(2).Color = 'k';
h3(2).Color = 'b';
clear h1 h2 h3;

subplot(2,3,3),
h11=histfit(mdl1r(3,:),bins,'kernel');hold on;
h22=histfit(mdl1n(3,:),bins,'kernel');xlim([0 2.3]);
h33=histfit(mdl1l(3,:),bins,'kernel');title('C4: AR{1}');
h11(1).Visible = 'off';
h22(1).Visible = 'off';
h33(1).Visible = 'off';
h11(2).Color = 'r';
h22(2).Color = 'k';
h33(2).Color = 'b';
clear h11 h22 h33;
%}
%%{
binnum = 20;
figure,
subplot(1,3,1),
histogram(mdl2r(1,:),'Normalization','probability','Numbins',binnum); hold on; 
histogram(mdl2n(1,:),'Normalization','probability','Numbins',binnum);title('C3');
histogram(mdl2l(1,:),'Normalization','probability','Numbins',binnum);

subplot(1,3,2),
histogram(mdl2r(2,:),'Normalization','probability','Numbins',binnum); hold on; 
histogram(mdl2n(2,:),'Normalization','probability','Numbins',binnum);title('Cz');
histogram(mdl2l(2,:),'Normalization','probability','Numbins',binnum);

subplot(1,3,3),
histogram(mdl2r(3,:),'Normalization','probability','Numbins',binnum); hold on; 
histogram(mdl2n(3,:),'Normalization','probability','Numbins',binnum);title('C4');
histogram(mdl2l(3,:),'Normalization','probability','Numbins',binnum);
%}
%% boxplot of the above
boxplot([mdl2r(1,1:281)',mdl2n(1,1:281)',mdl2l(1,1:281)']);

%% fit ARIMA model to the evolution of dynamics of DFA exponent -06/03/2018
%M is model, only identify the model order
%% fit ARMA model to the evolution of dynamics of DFA exponent
%M is model
for ch = 1:size(dfaR(:,:,:),1)
    for tr=1:size(dfaR(:,:,:),2)
        
        [M.DFA_r(ch,tr).p,M.DFA_r(ch,tr).q,M.DFA_r(ch,tr).mdl,M.DFA_r(ch,tr).LogL,M.DFA_r(ch,tr).stdr, ...
         M.DFA_r(ch,tr).E,M.DFA_r(ch,tr).h,M.DFA_r(ch,tr).t,M.DFA_r(ch,tr).z, ...
         M.DFA_r(ch,tr).YF,M.DFA_r(ch,tr).YMSE]= ARMA_for_DFA_exponents(squeeze(dfaR(ch,tr,:)));
       
        [M.DFA_n(ch,tr).p,M.DFA_n(ch,tr).q,M.DFA_n(ch,tr).mdl,M.DFA_n(ch,tr).LogL,M.DFA_n(ch,tr).stdr, ...
         M.DFA_n(ch,tr).E,M.DFA_n(ch,tr).h,M.DFA_n(ch,tr).t,M.DFA_n(ch,tr).z, ...
         M.DFA_n(ch,tr).YF,M.DFA_n(ch,tr).YMSE]= ARMA_for_DFA_exponents(squeeze(dfaN(ch,tr,:)));
       
        [M.DFA_l(ch,tr).p,M.DFA_l(ch,tr).q,M.DFA_l(ch,tr).mdl,M.DFA_l(ch,tr).LogL,M.DFA_l(ch,tr).stdr, ...
         M.DFA_l(ch,tr).E,M.DFA_l(ch,tr).h,M.DFA_l(ch,tr).t,M.DFA_l(ch,tr).z, ...
         M.DFA_l(ch,tr).YF,M.DFA_l(ch,tr).YMSE]= ARMA_for_DFA_exponents(squeeze(dfaL(ch,tr,:)));
       
     disp([ch,tr])
    end
end
%%
  for ch = 1:size(M.DFA_r,1)
      for tr = 1: 480%size(M.DFA_r,2)
          H_r(ch,tr)=M.DFA_r(ch,tr).h;
          H_n(ch,tr)=M.DFA_n(ch,tr).h;
          H_l(ch,tr)=M.DFA_l(ch,tr).h;
          
          p1_r(ch,tr)=M.DFA_r(ch,tr).mdl.AR{1};
          p1_n(ch,tr)=M.DFA_n(ch,tr).mdl.AR{1};
          p1_l(ch,tr)=M.DFA_l(ch,tr).mdl.AR{1};
          
          p2_r(ch,tr)=M.DFA_r(ch,tr).mdl.AR{2};
          p2_n(ch,tr)=M.DFA_n(ch,tr).mdl.AR{2};
          p2_l(ch,tr)=M.DFA_l(ch,tr).mdl.AR{2};
          
          q1_r(ch,tr)=M.DFA_r(ch,tr).mdl.MA{1};
          q1_n(ch,tr)=M.DFA_n(ch,tr).mdl.MA{1};
          q1_l(ch,tr)=M.DFA_l(ch,tr).mdl.MA{1};
      end
  end
  %%
  H = [reshape(H_r,1,numel(H_r)),reshape(H_n,1,numel(H_n)),reshape(H_n,1,numel(H_n))];
  histogram(reshape(H_r,1,numel(H_r)),'Normalization','probability');
  %% All the cases parameter values
  col = 3;
  H  = [reshape(p1_r,1,numel(p1_r)),reshape(p1_n,1,numel(p1_n)),reshape(p1_l,1,numel(p1_l))];
 subplot(1,col,1), histogram(H,'Normalization','probability');title('AR{1}');
  H2  = [reshape(p2_r,1,numel(p2_r)),reshape(p2_n,1,numel(p2_n)),reshape(p2_l,1,numel(p2_l))];
   subplot(1,col,2),histogram(H2,'Normalization','probability');title('AR{2}');
     H3  = [reshape(q1_r,1,numel(q1_r)),reshape(q1_n,1,numel(q1_n)),reshape(q1_l,1,numel(q1_l))];
   subplot(1,col,3),histogram(H3,'Normalization','probability');title('MA{1}');
  %% Right, Left, Neutral parameter values
 subplot(1,col,1), histogram(reshape(p1_r,1,numel(p1_r)),'Normalization','probability');title('AR{1}, Right (Blue), Neutral (red),Left(yellow)'); hold on; 
                 histogram(reshape(p1_n,1,numel(p1_n)),'Normalization','probability');
                 histogram(reshape(p1_l,1,numel(p1_l)),'Normalization','probability');
 subplot(1,col,2), histogram(reshape(p2_r,1,numel(p2_r)),'Normalization','probability');title('AR{2}, Right (Blue), Neutral (red),Left(yellow)'); hold on; 
                 histogram(reshape(p2_n,1,numel(p2_n)),'Normalization','probability');
                 histogram(reshape(p2_l,1,numel(p2_l)),'Normalization','probability');
 subplot(1,col,3), histogram(reshape(q1_r,1,numel(q1_r)),'Normalization','probability');title('AR{2}, Right (Blue), Neutral (red),Left(yellow)'); hold on; 
                 histogram(reshape(q1_n,1,numel(q1_n)),'Normalization','probability');
                 histogram(reshape(q1_l,1,numel(q1_l)),'Normalization','probability');
  %% channel wise distribution Right, Left, Neutral parameter values
    ch = 1;
 subplot(1,col,1), histogram(reshape(p1_r(ch,:),1,numel(p1_r(ch,:))),'Normalization','probability');title('C3 AR{1}, Right(Blue), Neutral(red),Left(yellow)'); hold on; 
                 histogram(reshape(p1_n(ch,:),1,numel(p1_n(ch,:))),'Normalization','probability');
                 histogram(reshape(p1_l(ch,:),1,numel(p1_l(ch,:))),'Normalization','probability');
 subplot(1,col,2), histogram(reshape(p2_r(ch,:),1,numel(p2_r(ch,:))),'Normalization','probability');title('C3 AR{2}, Right(Blue), Neutral(red),Left(yellow)'); hold on; 
                 histogram(reshape(p2_n(ch,:),1,numel(p2_n(ch,:))),'Normalization','probability');
                 histogram(reshape(p2_l(ch,:),1,numel(p2_l(ch,:))),'Normalization','probability');
 subplot(1,col,3), histogram(reshape(q1_r(ch,:),1,numel(q1_r(ch,:))),'Normalization','probability');title('C3 MA{1}, Right(Blue), Neutral(red),Left(yellow)'); hold on; 
                 histogram(reshape(q1_n(ch,:),1,numel(q1_n(ch,:))),'Normalization','probability');
                 histogram(reshape(q1_l(ch,:),1,numel(q1_l(ch,:))),'Normalization','probability');
     
   