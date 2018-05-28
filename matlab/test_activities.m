opts = detectImportOptions('four_activities.csv')
filename = readtable('four_activities.csv', opts);


for i=1:width(filename)
x = filename.Properties.VariableNames(i);
eval(sprintf('%s = filename.%s', x{1}, x{1}))
end

Y = y;
%square = Y .* Y;
X_x = Y;
St = 60;
F = 120; %batch_size x2 par rapport au St pour déclencher l'overlap 

%iter = length(X)/Fs
%e=Fs/2;%overlap
Z_x = [];
W_x = [];
Sd_x = [];
min_x = [];
max_x = [];
energy_sig_x = [];
transfo = [];

%for loop pour récupérer la moyenne et la variance, inscription dans des
%vecteurs // rejet de l'erreur avec catch

for i = 1:St:length(X_x)
sum1_x = 0;
sum2_x = 0;
try
    mini_x = min(X_x(i:i+F));
    min_x = [min_x ; mini_x];
    maxi_x = max(X_x(i:i+F));
    max_x = [max_x ; maxi_x];
    sum1_x = sum(X_x(i:i+F));
    M_x = sum1_x / length(X_x(i:i+F));
    Z_x = [Z_x ; M_x];
    sum2_x = var(X_x(i:i+F),1,'omitnan');
    V_x = sum2_x/length(X_x(i:i+F));
    W_x = [W_x ; V_x];
    %extraction des features : min, max, std, energy
    S_x = sqrt(V_x);
    Sd_x = [Sd_x ; S_x];
    %Ex_x = X_x(i:i+F)' * X_y(i:i+F);
    %Ex_x = sum(conj(X_x(i:i+F)) .* X_x(i:i+F));
    Ex_x = sum(abs(X_x(i:i+F)).^2);
    energy_sig_x = [energy_sig_x ; Ex_x];
    
catch
    disp('index exceeds matrix dimension')
end

end

Y = y;
%square = Y .* Y;
X_y = Y;

Z_y = [];
W_y = [];
Sd_y = [];
min_y = [];
max_y = [];
energy_sig_y = [];
test = []

for i = 1:St:length(X_y)
sum1_y = 0;
sum2_y = 0;
try
    
    %rootmsq = rms(gFx(i:i+F) + gFy(i:i+F) + gFz(i:i+F));
    %rootms = [rootms ; rootmsq];
    mini_y = min(X_y(i:i+F));
    min_y = [min_y ; mini_y];
    maxi_y = max(X_y(i:i+F));
    max_y = [max_y ; maxi_y];
    sum1_y = sum(X_y(i:i+F));
    M_y = sum1_y / length(X_y(i:i+F));
    Z_y = [Z_y ; M_y];
    sum2_y = var(X_y(i:i+F),1,'omitnan');
    V_y = sum2_y/length(X_y(i:i+F));
    W_y = [W_y ; V_y];
    %extraction des features : min, max, std, energy
    S_y = sqrt(V_y);
    Sd_y = [Sd_y ; S_y];
    %Ex_y = X_y(i:i+F)' * X_y(i:i+F);
    %Ex_y = sum(conj(X_y(i:i+F)) .* X_y(i:i+F));
    Ex_y = sum(abs(X_y(i:i+F)).^2);
    energy_sig_y = [energy_sig_y ; Ex_y];
    
    %test_y = fft(X_y(i:i+F))
    %test_sig = sum(abs(test_y).^2)/length(test_y)
    %test = [test; test_sig]
    
catch
    disp('index exceeds matrix dimension')
end

end
Y = z;
%square = Y .* Y;
X_z = Y;

Z_z = [];
W_z = [];
Sd_z = [];
min_z = [];
max_z = [];
energy_sig_z = [];

for i = 1:St:length(X_z)
sum1_z = 0;
sum2_z = 0;
try
    mini_z = min(X_z(i:i+F));
    min_z = [min_z ; mini_z];
    maxi_z = max(X_z(i:i+F));
    max_z = [max_z ; maxi_z];
    sum1_z = sum(X_z(i:i+F));
    M_z = sum1_z / length(X_z(i:i+F));
    Z_z = [Z_z ; M_z];
    sum2_z = var(X_z(i:i+F),1,'omitnan');
    V_z = sum2_z/length(X_z(i:i+F));
    W_z = [W_z ; V_z];
    %extraction des features : min, max, std, energy
    S_z = sqrt(V_z);
    Sd_z = [Sd_z ; S_z];
    %Ex_z = X_z(i:i+F)' * X_z(i:i+F);
    %Ex_z = sum(conj(X_z(i:i+F)) .* X_z(i:i+F));
    Ex_z = sum(abs(X_z(i:i+F)).^2);
    energy_sig_z = [energy_sig_z ; Ex_z];
    
catch
    disp('index exceeds matrix dimension')
end

end

%rejet des valeurs nulles, pour éviter les erreurs
sum1_x = sum1_x(sum2_x>0);
sum2_x = sum2_x(sum2_x>0);
M_x = M_x(M_x>0);
Z_x = Z_x(Z_x>0);
V_x = V_x(V_x>0);
W_x = W_x(W_x>0);

%[B,A] = findpeaks(W_x,'MinPeakHeight',0.000000286,'MaxPeakWidth', 5, 'Annotate', 'extents')
% récupération des index des pics avec paramètres de hauteur des pics,
% largeur

%repérage des moments d'activité : marche, debout statique
counter = 0
%recueillir les phases d'activité - inactivité sur un autre axe
%for i = 1:length(A)-1
%    if (A(i+1) - A(i)) <=6  && counter ~= 1
%        disp('OK à')
%        disp(A(i))
%        counter = 1;
%    elseif (A(i+1) - A(i)) <= 6 && counter == 1
%        disp('STOP à')
%        disp(A(i))
%        counter = counter - 1;
%    end
%end

figure;
plot(W_x) %Variance
title('Variance')
figure;
plot(W_y)
figure;
plot(W_z)
%figure;
%plot(Z_x) %Moyenne
%title("Moyenne")
%fprintf('Variance avec batch_size = %i valeurs et overlap de 50%\n', F)

%for i=1:length(W)
%    if W(i)>0.0002
%        fprintf('la marche a commencé à %f',W(i))
%    end
%    if W(i) <0.0002
%        fprintf("la marche s'est terminée à %f", W(i))
%    end
%end
%figure;
%plot(time,gFz)
%hold on
%plot(linspace(0,195,length(W_y)), W_y,'r') %plot des courbes de moyenne et
%signal brut pour évaluer la justesse des calculs

%figure;
%hold off
%plot(energy_sig_x)
%title("énergie du signal")

fid = fopen('features_x119_window119.csv','w'); 
fprintf(fid,'%s\n',['min,max,std,energy_signal'])
fclose(fid)
%write data to end of file
dlmwrite('features_x119_window119.csv',[min_x,max_x,Sd_x,energy_sig_x],'-append');

fid = fopen('features_y119_window119.csv','w'); 
fprintf(fid,'%s\n',['min,max,std,energy_signal'])
fclose(fid)
%write data to end of file
dlmwrite('features_y119_window119.csv',[min_z,max_z,Sd_z,energy_sig_z],'-append');

fid = fopen('features_z119_window119.csv','w'); 
fprintf(fid,'%s\n',['min,max,std,energy_signal'])
fclose(fid)
%write data to end of file
dlmwrite('features_z119_window119.csv',[min_z,max_z,Sd_z,energy_sig_z],'-append');


%try
%    column_names = {'min_x', 'max_x', 'Sd', 'energy_sig_x'};
%    v2 = min_x ; v3 = max_x ; v4 = Sd ; v5 = energy_sig_x ;
%    fid = fopen('featureX_ellcie.txt','wt');
%    fprintf(fid, '%s ', column_names{:});
%    fprintf(fid, '\n');
%    block_of_data = [v2, v3, v4, v5];
%    fmt = repmat('%15g ', 1, 4);
%    fmt(end:end+1) = '\n';
%    fprintf(fid, fmt, block_of_data.');   %transpose is needed to get everything right
%    fclose(fid);
%catch 
%    fprintf('dimensions of matrices being concatenated are not consistent')
%end