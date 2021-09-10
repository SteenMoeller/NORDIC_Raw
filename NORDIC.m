function  NORDIC(file_filter,ARG)
%
%   Processing of matlab files. Assume that the data is a 4D array, and
%   that the 4th dimension has [vol_1 ...   Vol_N  noise_volume gfactor_map ]
%   Variable should be stored in an array labelled KSP
%   KSP should be in image-space and complex valued
%   An auxilary variable called KSP_processed shoudl exist. It shoudl be
%   all zeros, and have the length of the third dimension in KSP


NORDIC_threshold=1.0;         %  scaling the threshold relative to the largest singular value of the noise
ker=0;                        %  if 0, the scaling is automatic to a 11:1 ratio
kernel_size=[ker ker ker];    %  making a cubic patchsize
master_fast=1;                %  If 0, then indidvidual files will be saved. legacy option, not needed when patch_avg_sub<ker
soft_thrs=[];                 %  Parameter for switching between NORDIC and MPPCA  (=10) 
patch_avg_sub=2;              %  use for master_fast==1;  Reduces the number of over lapping patches. Shift us 1/patch_avg_sub
patch_avg=0;                  %  For subfunction use, of where patch averagign shoudl be performed.  
file_idx=1; 
DUAL=1;                       %  Paramter to remove some final volumes from the end of [vol_1 ...   Vol_N] leaving [ noise_volume gfactor_map ]
restart_list=0;
DIR=[pwd '/'];
                
NVR_LLR_3D_MP_2(DIR,file_filter,file_idx,NORDIC_threshold,DUAL,kernel_size,patch_avg,['kernel_' num2str(ker)],restart_list, master_fast,soft_thrs, patch_avg_sub);


return


function  NVR_LLR_3D_MP_2(DIREC,file_filter,file_idx,NVR_threshold,DUAL,kernel_size,patch_average,APPEND_NAME, restart_list, master_fast,soft_thrs, patch_average_sub, NVR2, truncate_length)


% check if the NVR file exist, otherwise start the conversion
%       there are three files,     filename, filename_out   filename_in
% check if the data is in the file; otherwise skip 
% check if the file is being handled allready, otherwise only one slice can be processed
%
% start processing
%   check if the output is saved in a file, otherwise start processing
% finish processing
%    save the result to a file
%    clear all temporary files   
%    if complete then load the file and do the normalization for averaging
    
   % path('~moeller/matlab/Regularization',path)
 
%soft_thrs=10;  disp('Using the MPPCA estimation method')
% soft_thrs=[];  disp('Using the NVR estimation method')
   

if ~exist('master_fast'); master_fast=0;   end;  
if ~exist('patch_average_sub'); ARG.patch_average_sub=1; else ;  ARG.patch_average_sub=patch_average_sub;  end

if ~exist('DUAL'); ARG.DUAL=1; else ;  ARG.DUAL=DUAL;  end
if ~exist('NVR_threshold'); ARG.NVR_threshold=1; else ;  ARG.NVR_threshold=NVR_threshold;    end
if ~exist('file_idx'); ARG.file_idx=1; else ;  ARG.file_idx= file_idx;  end
if ~exist('kernel_size'); ARG.kernel_size=[4 4 4];else ;  ARG.kernel_size= kernel_size;  end
if ~exist('patch_average'); ARG.patch_average=1; else ;  ARG.patch_average=patch_average;  end
if ~exist('APPEND_NAME');  ARG.APPEND_NAME=[];else ;  ARG.APPEND_NAME=APPEND_NAME;   end
if ~exist('NVR2');  ARG.NVR2=[]; else ;  ARG.NVR2=NVR2;  end
if ~exist('truncate_length');  ARG.truncate_length=[]; else ;  ARG.truncate_length= truncate_length; end
if ~exist('restart_list');  ARG.restart_list=0;else ;  ARG.restart_list= restart_list;  end
if ~exist('soft_thrs');  ARG.soft_thrs=[] ;else ;  ARG.soft_thrs= soft_thrs;  end

if isempty(soft_thrs)
     disp('Using the NVR estimation method')
elseif soft_thrs==10
     disp('Using the MPPCA estimation method')
end

ARG.patch_scale=0;

PP=dir(file_filter) % PP=dir('M*SORT2*mat');
filename=PP(file_idx).name;%filename2%[1:2]+2+2+2%size(PP,1)/2:-1:1;%:-1:size(PP,1)/2
if sum(kernel_size)==0  % choose optimal kernel
tmp=matfile(filename);
matdim=size(tmp,'KSP');
ker=round((matdim(4)*11)^(1/3));
kernel_size=[ker ker ker], ARG.kernel_size=kernel_size;
APPEND_NAME=['kernel' num2str(ker)];  ARG.APPEND_NAME=APPEND_NAME; 
end



ARG.filename=filename;
ARG.master_fast=master_fast;
  filename_out=dir(['KSP_' filename(1:end-4)   APPEND_NAME  '.mat']);  % FILE with the NVR reconstruction
  filename_in=dir([filename(1:end-4)     '_NVR.mat']);   % file with the data that will be used in the NVR, IE phase normalized

if isempty(filename_out) |1  % process is not complete
  
  if isempty(filename_in) | 1 %  Initial file is missing, create
      
   %  STANDARD 
   disp('CHECK if the right input to NORDIC is used')
    [KSP_phase_norm, meanphase, DD_phase] = sub_create_NVR_file(filename, ARG) ;
   %  [KSP_phase_norm, meanphase, DD_phase] = sub_create_NVR_file_HCP(filename, ARG) ;
   %  [KSP_phase_norm, meanphase, DD_phase] = sub_create_NVR_file_MPPCA(filename, ARG) ;
    
     ARG.meanphase=meanphase;
     ARG.DD_phase=DD_phase;
       filename_in=dir([filename(1:end-4)     '_NVR.mat']); 
  QQ=matfile(filename_in.name,'writable',true);
 % return
      
      for n1=1:1000;   % update with the files that already exist
          tmp=dir([ARG.filename  'slice' num2str(n1)  '.mat']);
          if ~isempty(tmp);  QQ.KSP_processed(1,n1)=2; end
      end
  else
      return;  % Only the master gets to go. Usefull when the phase correction and saving is the slowest part
  end
  
  if ~isempty(filename_in) % check if it is being written to or if it is completed
       filename_in_tmp=dir([filename(1:end-4)     '_NVR.mat']);  
     if filename_in_tmp.bytes ~= filename_in.bytes  | filename_in_tmp.bytes<1e3
         disp('NVR file is being saved or processed. Stopping....')
       %  return
     end
  end
  
  if restart_list~=2 
  filename_in=dir([filename(1:end-4)     '_NVR.mat']); 
  QQ=matfile(filename_in.name,'writable',true);
  
  if master_fast==1; QQ.writing_file=1; end
  
  if QQ.writing_file==0
      disp('I am a slave, just saving files')
  end
  end
  
  
  if restart_list==3 
      %QQ.writing_file=1; % Make this the master
      KSP_processed=QQ.KSP_processed;
      if sum(KSP_processed)==2*size(KSP_processed,2) % all files are saved
          QQ.writing_file=1;  
      else
          KSP_processed(KSP_processed~=2)=0;
          QQ.KSP_processed=KSP_processed;
          
       %   return
      end
      
      for n1=1:1000;   % update with the files that already exist
          tmp=dir([ARG.filename  'slice' num2str(n1)  '.mat']);
          if ~isempty(tmp);  QQ.KSP_processed(1,n1)=2; end
      end
      
      
  end
  
  
  
  if restart_list==2  % the _NVR file has been corrupted
    %  QQ.writing_file=0; % lock file for other to be maste    
      sub_create_NVR_file(filename, ARG)
    filename_in=dir([filename(1:end-4)     '_NVR.mat']); 
  QQ=matfile(filename_in.name,'writable',true);
  
      
      for n1=1:1000;   % update with the files that already exist
          tmp=dir([ARG.filename  'slice' num2str(n1)  '.mat']);
          if ~isempty(tmp);  QQ.KSP_processed(1,n1)=2; end
      end
      restart_list=1;
  end
  
 
  
  if restart_list==1 
      QQ.writing_file=1; % Make this the master
      KSP_processed=QQ.KSP_processed;
      KSP_processed(KSP_processed==1)=0;
      KSP_processed(KSP_processed==3)=0;
      QQ.KSP_processed=KSP_processed;
  end
  
  
   if QQ.writing_file==1
      disp('I am master, I have control')
      QQ.writing_file=0;
      %KSP2=QQ.KSP2;
     variableInfo=who('-file',filename_in.name);
        if  (ismember('KSP_recon',variableInfo) & restart_list==0) 
           KSP_recon=QQ.KSP_recon;
        elseif ismember('size_KSP',variableInfo)
            size_KSP=QQ.size_KSP;
            KSP_recon=single(complex(zeros(size_KSP)));
            KSP2=KSP_recon;
        else
           KSP_recon=single(complex(zeros(size(QQ,'KSP2'))));
           KSP2=KSP_recon;
        end
      master=1;
   else
      %  disp('I am slave')
      KSP2=[];
      KSP_recon=[];
      master=0;
     
   end
   
   if restart_list==1;
       %QQ.writing_file==1
       if sum(QQ.KSP_processed)==size(QQ.KSP_processed,2)*3;
           QQ.KSP_processed=QQ.KSP_processed*0+2; % set all files to be processed. If they are not they will be.
           KSP_recon=single(complex(zeros(size(KSP2))));
       end
       
       for n1=1:1000;   % update with the files that already exist
           tmp=dir([ARG.filename  'slice' num2str(n1)  '.mat']);
           if ~isempty(tmp);
               QQ.KSP_processed(1,n1)=2;
         %  else
         %      QQ.KSP_processed(1,n1)=0;
           end
       end       
   end
   
   if ARG.patch_average==0
       
       KSP_processed=QQ.KSP_processed*0;
       for nw1=2:max(1,floor(ARG.kernel_size(1)/ARG.patch_average_sub));
       KSP_processed(1,nw1 : max(1,floor(ARG.kernel_size(1)/ARG.patch_average_sub)):end)=2;
       end
       KSP_processed(end)=0; % disp
       QQ.KSP_processed=KSP_processed;    
   end
  
   if master_fast>0;
       KSP2=KSP_phase_norm;
     % KSP2=QQ.KSP2; 
   end
   
   if master_fast==0
       KSP_weight=KSP2(:,:,:,1)*0;
   for n1=1:size(QQ,'KSP_processed',2)
       fprintf( [num2str(n1) ' '])
     [KSP_recon]=sub_NVR_for_one_slice(KSP_recon,KSP2,ARG,n1,QQ,master*0,[])   ;    % save all files
       fprintf( [num2str(n1) ' '])
    if mod(n1,20)==0; disp(' ');end
   end
   elseif master_fast==1
       
       KSP_weight=KSP2(:,:,:,1)*0;
       NOISE=KSP_weight;
       Component_threshold=KSP_weight;
       energy_removed=KSP_weight;
       
    for n1=1:size(QQ,'KSP_processed',2)
       fprintf( [num2str(n1) ' '])
     [KSP_recon,~,KSP_weight,NOISE,Component_threshold,energy_removed]=sub_NVR_for_one_slice(KSP_recon,KSP2,ARG,n1,QQ,master_fast,KSP_weight,NOISE,Component_threshold,energy_removed)   ;    % save all files
       fprintf( [num2str(n1) ' '])
    if mod(n1,20)==0; disp(' ');end
    end
    
     KSP_recon=KSP_recon./repmat((KSP_weight),[1 1 1 size(KSP2,4)]);  % Assumes that the combination is with N instead of sqrt(N). Works for NVR not MPPCA
     ARG.NOISE=  NOISE./KSP_weight;
     ARG.Component_threshold = Component_threshold./KSP_weight;
     ARG.energy_removed = energy_removed./KSP_weight;
     
    elseif master_fast==2
        
       KSP_processed=QQ.KSP_processed;   
       
       KSP_weight=KSP2(:,:,:,1)*0;
       KSP_recon2=KSP_recon;
    for n1=1:size(QQ,'KSP_processed',2)
       fprintf( [num2str(n1) ' '])
     [KSP_recon2,~,KSP_weight]=sub_NVR_for_one_slice(KSP_recon2,abs(KSP2),ARG,n1,QQ,master_fast*0+1,KSP_weight)   ;    % save all files
       fprintf( [num2str(n1) ' '])
    if mod(n1,20)==0; disp(' ');end
    end
    
     KSP_recon2=KSP_recon2./repmat((KSP_weight),[1 1 1 size(KSP2,4)]);  % Assumes that the combination is with N instead of sqrt(N). Works for NVR not MPPCA
      
     QQ.KSP_processed=KSP_processed;   
     
       KSP_weight=KSP2(:,:,:,1)*0;       
    for n1=1:size(QQ,'KSP_processed',2)
       fprintf( [num2str(n1) ' '])
     [KSP_recon,~,KSP_weight]=sub_NVR_for_one_slice(KSP_recon,abs(KSP_recon2).*exp(1i*angle(KSP2)),ARG,n1,QQ,master_fast*0+1,KSP_weight)   ;    % save all files
       fprintf( [num2str(n1) ' '])
    if mod(n1,20)==0; disp(' ');end
    end
       KSP_recon=KSP_recon./repmat((KSP_weight),[1 1 1 size(KSP2,4)]);  % Assumes that the combination is with N instead of sqrt(N). Works for NVR not MPPCA
     
   elseif master_fast==3
       
       KSP_weight=KSP2(:,:,:,1)*0;
    for n1=1:size(QQ,'KSP_processed',2)
       fprintf( [num2str(n1) ' '])
     [KSP_recon,~,KSP_weight]=sub_NVR_for_one_slice(KSP_recon,abs(KSP2),ARG,n1,QQ,master_fast*0+1,KSP_weight)   ;    % save all files
       fprintf( [num2str(n1) ' '])
    if mod(n1,20)==0; disp(' ');end
    end
    
     KSP_recon=KSP_recon./repmat((KSP_weight),[1 1 1 size(KSP2,4)]);  % Assumes that the combination is with N instead of sqrt(N). Works for NVR not MPPCA
                   
   end
   
   
  
   if master_fast==0;  % allready loaded
  if master==1
      KSP_processed=QQ.KSP_processed;
      if sum(KSP_processed(KSP_processed<2))>0;pause(60*5);end  % hopefully enough time for other processes to finish
   for n1=1:size(QQ,'KSP_processed',2)
       fprintf( [num2str(n1) ' '])
     [KSP_recon]=sub_NVR_for_one_slice(KSP_recon,KSP2,ARG,n1,QQ,master)   ;  % have the master load the files in  
       fprintf( [num2str(n1) ' '])
    if mod(n1,20)==0; disp(' ');end
   end
  
  end 
   end
  % detrmine if this will be the master or slave process
  
  
  % GOTTA SAVE THE RESULTS, and that goes into the *_NVR file
  if restart_list==0
  if master==1  & master_fast==0
      QQ.KSP_recon=KSP_recon;
  end
  end
  
  
 
  
  if master==1
  if sum(QQ.KSP_processed,2)==3*size(QQ.KSP_processed,2) % all slices processed and included
     disp('Complete process and save new file')
     sub_complete_NVR(KSP_recon,QQ,ARG)
  elseif master_fast>0
       sub_complete_NVR(KSP_recon,QQ,ARG);   % run in a single_file and DOES NOT average
  else
      disp('something went wrong for the master. Check ')
      QQ.KSP_processed
      sum(QQ.KSP_processed,2)
      3*size(QQ.KSP_processed,2)
  end
  
  end
  
   if master==1
   % cleanup
        filename_out=dir(['KSP_' filename(1:end-4)   APPEND_NAME  '.mat']);  % FILE with the NVR reconstruction
     if ~isempty(filename_out)
        sub_remove_tmp_files(ARG) 
     end
        
        
   end
  

end

function  sub_remove_tmp_files(ARG) 
 filename_in=dir([ARG.filename(1:end-4)     '_NVR.mat']);   % file with the data that will be used in the NVR, IE phase normalized
eval(['!rm '  filename_in.name])
 
 filename_in=dir([ARG.filename(1:end)     'slice*.mat']);   % file with the data that will be used in the NVR, IE phase normalized
for n=1:size(filename_in,1)
    eval(['!rm ' filename_in(n).name  ]);   
end
    

function  sub_complete_NVR(KSP2a,QQ,ARG)

        KSP2_weight_update=0*KSP2a(:,:,:,1);
        
filename=ARG.filename;
patch_average = ARG.patch_average;
APPEND_NAME   = ARG.APPEND_NAME;
NVR_threshold = ARG.NVR_threshold;
    w1=ARG.kernel_size(1);
    w2=ARG.kernel_size(2);
    w3=ARG.kernel_size(3);
      
% create  KSP2_weight       
if    patch_average==1
    if ~exist('patch_avg'); patch_avg=1;end
    KSP2_weight=zeros((size(KSP2a(1:w1,:,:,1))));
    for n2=1:size(KSP2a,2)-w2;
        for n3=1:size(KSP2a,3)-w3;
            if patch_avg==1
                KSP2_weight(:,[1:w2]+(n2-1),[1:w3]+(n3-1),1) =...
                    KSP2_weight(:,[1:w2]+(n2-1),[1:w3]+(n3-1),1) +1;
            else
                KSP2_weight(:,round(w2/2)+(n2-1),round(w3/2)+(n3-1),1) =...
                    KSP2_weight(:,round(w2/2)+(n2-1),round(w3/2)+(n3-1),1) +1;
            end
        end
    end
    
    for n1= 1:size(KSP2a,1)-w1;
        KSP2_weight_update([1:w1]+(n1-1),:,:,1)=KSP2_weight_update([1:w1]+(n1-1),:,:,1)+ KSP2_weight;
    end
    
    KSP_update=KSP2a./repmat(KSP2_weight_update,[1 1 1 size(KSP2a,4)]);
else
    KSP_update=KSP2a;
end


if ARG.master_fast~=3
    QQ1=matfile(ARG.filename);
    g_factor_map=QQ1.KSP(:,:,:,size(QQ1,'KSP',4));
    
    for niter=1:min(size(KSP_update,4),size(ARG.DD_phase,4)    )
        KSP_update(:,:,:,niter)=  KSP_update(:,:,:,niter).*exp(1i*angle(ARG.meanphase )).*exp(1i*angle(ARG.DD_phase(:,:,:,niter) ));
    end
    for niter=1:size(KSP_update,4); KSP_update(:,:,:,niter)=KSP_update(:,:,:,niter).*(g_factor_map+eps); end
end


try
 NOISE= ARG.NOISE;  if ARG.soft_thrs~=10; NOISE=[]; end  % empty it unless it is calculated
Component_threshold= ARG.Component_threshold;
energy_removed = ARG.energy_removed;
catch;end
KSP_update(isnan(KSP_update))=0;
KSP_update(isinf(KSP_update))=0;


              % KSP_update=DATA_full;
    %  for niter=1:size(KSP_update,4); KSP_update(:,:,:,niter)=KSP_update(:,:,:,niter).*(g_factor_map+eps); end
      
      
      try
        if isempty(APPEND_NAME)
           save(['KSP_' filename(1:end-4)  '.mat'],'KSP_update','Component_threshold','energy_removed','NOISE','-v7.3')
          else
           save(['KSP_' filename(1:end-4)   APPEND_NAME  '.mat'],'KSP_update','Component_threshold','energy_removed','NOISE','-v7.3')         
        end
      catch
        if isempty(APPEND_NAME)
           save(['KSP_' filename(1:end-4)  '.mat'],'KSP_update','-v7.3')
          else
           save(['KSP_' filename(1:end-4)   APPEND_NAME  '.mat'],'KSP_update','-v7.3')         
        end
     
      end
      
      
          
return















function  [KSP_recon,KSP2,KSP2_weight,NOISE, Component_threshold,energy_removed]=sub_NVR_for_one_slice(KSP_recon,KSP2,ARG,n1,QQ,master,KSP2_weight,NOISE,Component_threshold,energy_removed)

if ~exist('NOISE'); NOISE=[];  end
if ~exist('Component_threshold');Component_threshold=[];  end
if ~exist('energy_removed');  energy_removed=[]; end
%  QQ.KSP_processed  0 nothing done, 1 running, 2 saved 3 completed and  averaged

if master==0 && ARG.patch_average==0
  OPTION='NO_master_NO_PA';
  
elseif master==0 && ARG.patch_average==1
  OPTION='NO_master_PA'    ;

elseif master==1 && ARG.patch_average==0
  OPTION='master_NO_PA'    ;
 
elseif master==1 && ARG.patch_average==1
  OPTION='master_PA'   ; 
  
end
    
switch OPTION
    
    case 'NO_master_NO_PA'
        
    case  'NO_master_PA'
        
    case  'master_NO_PA'
        
    case  'master_PA'
        
end


if     QQ.KSP_processed(1,n1)~=1  && QQ.KSP_processed(1,n1)~=3 % not being processed also not completed yet
    
    if     QQ.KSP_processed(1,n1)==2 && master==1%  processed but not added.
        % loading instead of processing
        % load file as soon as save, if more than 10 sec, just do the recon
        % instead.
        try  % try to load otherwise go to next slice
            load([ARG.filename  'slice' num2str(n1)  '.mat'],'DATA_full2')
        catch;
            QQ.KSP_processed(1,n1)=0;  % identified as bad file and being identified for reprocessing
            return  ;end
    end
    
    if QQ.KSP_processed(1,n1)~=2
        QQ.KSP_processed(1,n1)=1; % block for other processes
        if ~exist('DATA_full2')
            ARG2=QQ.ARG;
            if master==0
                QQ.KSP_processed(1,n1)=1    ;  % STARTING
                KSP2a=QQ.KSP2([1:ARG.kernel_size(1)]+(n1-1),:,:,:); lambda=ARG2.LLR_scale*ARG.NVR_threshold;
            else
                QQ.KSP_processed(1,n1)=1    ;  % STARTING
                KSP2a=KSP2([1:ARG.kernel_size(1)]+(n1-1),:,:,:); lambda=ARG2.LLR_scale*ARG.NVR_threshold;
                
            end
            
            if    ARG.patch_average==1
              %  [DATA_full2, ~,NOISE, Component_threshold] =subfunction_loop_for_NVR_avg(KSP2a,ARG.kernel_size(3),ARG.kernel_size(2),ARG.kernel_size(1),lambda,1,ARG.soft_thrs);
                 [DATA_full2, KSP2_weight] =subfunction_loop_for_NVR_avg(KSP2a,ARG.kernel_size(3),ARG.kernel_size(2),ARG.kernel_size(1),lambda,1,ARG.soft_thrs,KSP2_weight);
            else
                
                KSP2_weight_tmp         =KSP2_weight([1:ARG.kernel_size(1)]+(n1-1),:,:,:);                
                NOISE_tmp               =NOISE([1:ARG.kernel_size(1)]+(n1-1),:,:,:);
                Component_threshold_tmp =Component_threshold([1:ARG.kernel_size(1)]+(n1-1),:,:,:);
                energy_removed_tmp      =energy_removed([1:ARG.kernel_size(1)]+(n1-1),:,:,:);
                
                 [DATA_full2,KSP2_weight_tmp,NOISE_tmp, Component_threshold_tmp,energy_removed_tmp] =...
                     subfunction_loop_for_NVR_avg_update(KSP2a,ARG.kernel_size(3),ARG.kernel_size(2),ARG.kernel_size(1),lambda,1,ARG.soft_thrs,KSP2_weight_tmp,ARG,NOISE_tmp,Component_threshold_tmp,energy_removed_tmp);
                 
                KSP2_weight([1:ARG.kernel_size(1)]+(n1-1),:,:,:)=KSP2_weight_tmp;
                
            try;     NOISE([1:ARG.kernel_size(1)]+(n1-1),:,:,:) =NOISE_tmp;  catch;end
                Component_threshold([1:ARG.kernel_size(1)]+(n1-1),:,:,:) = Component_threshold_tmp;
                energy_removed([1:ARG.kernel_size(1)]+(n1-1),:,:,:)  = energy_removed_tmp;
                
                %DATA_full=subfunction_loop_for_NVR(KSP2a,ARG.kernel_size(3),ARG.kernel_size(2),ARG.kernel_size(1),lambda);
                %DATA_full2(1, round(w2/2)+[1:size(DATA_full,1)],:,:  )=DATA_full;  % center plane only
            end
            
        end
        
    end
    
    
    
    if master==0
        if QQ.KSP_processed(1,n1)~=2 
         save([ARG.filename  'slice' num2str(n1)  '.mat'],'DATA_full2', '-v7.3'        )
         QQ.KSP_processed(1,n1)=2    ;  % COMPLETED
        end
    else
        
        if    ARG.patch_average==1
            tmp=KSP_recon([1:ARG.kernel_size(1)]+(n1-1),:,:,:) ;
            KSP_recon([1:ARG.kernel_size(1)]+(n1-1),:,:,:)= tmp + DATA_full2;
        else
            KSP_recon([1:ARG.kernel_size(1)]+(n1-1) ,1:size(DATA_full2,2),:,:)=  KSP_recon([1:ARG.kernel_size(1)]+(n1-1) ,1:size(DATA_full2,2),:,:) +  DATA_full2;
        end
        QQ.KSP_processed(1,n1)=3     ;
    end
    
    
end


return



function [KSP2, meanphase, DD_phase]=sub_create_NVR_file(filename,ARG) 

tmp=1;
filename_in=([filename(1:end-4)     '_NVR.mat']);   % file with the data that will be used in the NVR, IE phase normalized
save(filename_in,'tmp');  % create nearly empty file

      
    newname=load(filename,'KSP');               
    matdim=size(newname.KSP(:,:,:,1:end-(ARG.DUAL-1)));  % just to indicate if the is noise + SB_refrence + gfactor  or noise +gfactor added
    
%    KSP=newname.KSP(:,:,:,1:end-(ARG.DUAL-1)); clear newname
KSP=newname.KSP;clear newname
    for nsl=size(KSP,3):-1:1
 %       MASK(:,:,nsl)= bwmorph( abs(KSP(:,:,nsl,1))>0,'shrink',1);
    end
    
    if ARG.DUAL==3 | ARG.DUAL==1  |  1
     g_factor_map=mean(KSP(:,:,:,end:end)  ,4)*1;
    else
     g_factor_map=mean(KSP(:,:,:,end-(ARG.DUAL-1):end)  ,4);
    end
    
    KSP=KSP(:,:,:,1:end-1-(ARG.DUAL-1));  % removes the gfactor and the SB reference if it is there.
    
    for niter=1:size(KSP,4); KSP(:,:,:,niter)=KSP(:,:,:,niter)./(g_factor_map+eps); end
    
   
    
   % D1=KSP(:,:,:,1); SNRMASK=abs(D1(:,:,:,end))<20;
   % KSP_NOISE=D1.*SNRMASK.*MASK.*sqrt(size(D1,1))*pi/2;
         % KSP_NOISE=D1.*SNRMASK.*MASK;%.*sqrt(size(D1,1))*pi/2;
    KSP_NOISE = KSP(:,:,:,end);
  %  for niter=1:size(KSP,4); KSP(:,:,:,niter)=KSP(:,:,:,niter).*SNRMASK.*MASK;end
  %  clear MASK SNRMASK
    
  
  
  meanphase=mean(KSP(:,:,:,[1:end-1]),4);
  
    for slice=matdim(3):-1:1       
        for n=1:size(KSP,4)-1; % include the noise 
            KSP(:,:,slice,n)=KSP(:,:,slice,n).*exp(-i*angle(meanphase(:,:,slice)));
        end
    end
        DD_phase=0*KSP;
        
   for slice=matdim(3):-1:1   
      for n=1:size(KSP,4)-1; 
         tmp=KSP(:,:,slice,n);
            for ndim=[1:2]; tmp=ifftshift(ifft(ifftshift( tmp ,ndim),[],ndim),ndim+0); end
            [nx, ny, nc, nb] = size(tmp(:,:,:,:,1,1));
            tmp = bsxfun(@times,tmp,reshape(tukeywin(ny,1).^10,[1 ny]));
            tmp = bsxfun(@times,tmp,reshape(tukeywin(nx,1).^10,[nx 1]));
            for ndim=[1:2]; tmp=fftshift(fft(fftshift( tmp ,ndim),[],ndim),ndim+0); end
            DD_phase(:,:,slice,n)=tmp;   
      end
   end
     % DD_phase=0*KSP;
    
     for slice=matdim(3):-1:1   
      for n=1:size(KSP,4); 
         KSP_update(:,:,slice,n)= KSP(:,:,slice,n).*exp(-i*angle( DD_phase(:,:,slice,n)   ));
      end
     end
     
        
        if 0  %  20200527
        
    for slice=matdim(3):-1:1
        DD=KSP(:,:,slice,:);
        meanphase=mean(DD(:,:,1,[1:size(DD,4)-1]),4);
        for n=1:size(DD,4); % include the noise 
            DD(:,:,1,n)=DD(:,:,1,n).*exp(-i*angle(meanphase));
        end
     %   for n=1:size(DD,4);
     %       tmp=DD(:,:,1,n);
     %       for ndim=[1:2]; tmp=ifftshift(ifft(ifftshift( tmp ,ndim),[],ndim),ndim+0); end
     %       [nx, ny, nc, nb] = size(tmp(:,:,:,:,1,1));
     %       tmp = bsxfun(@times,tmp,reshape(tukeywin(ny,1).^10,[1 ny]));
     %       tmp = bsxfun(@times,tmp,reshape(tukeywin(nx,1).^10,[nx 1]));
     %       for ndim=[1:2]; tmp=fftshift(fft(fftshift( tmp ,ndim),[],ndim),ndim+0); end
     %       DD_phase(:,:,1,n)=tmp;
     %   end
     %   meanphase=mean(DD(:,:,1,[1:size(DD,4)-1]),4);
     %   for n=1:size(DD,4)-1; % not added to noise
     %       DD(:,:,1,n)=DD(:,:,1,n).*exp(-i*angle(meanphase));
     %   end
        for n=1:size(DD,4);
            tmp=DD(:,:,1,n);
            for ndim=[1:2]; tmp=ifftshift(ifft(ifftshift( tmp ,ndim),[],ndim),ndim+0); end
            [nx, ny, nc, nb] = size(tmp(:,:,:,:,1,1));
            tmp = bsxfun(@times,tmp,reshape(tukeywin(ny,1).^10,[1 ny]));
            tmp = bsxfun(@times,tmp,reshape(tukeywin(nx,1).^10,[nx 1]));
            for ndim=[1:2]; tmp=fftshift(fft(fftshift( tmp ,ndim),[],ndim),ndim+0); end
            DD_phase(:,:,1,n)=tmp;
        end
        for n=1:size(DD,4);
            DD1(:,:,1,n)=DD(:,:,1,n).*exp(-i*angle( DD_phase(:,:,1,n)   ));
        end
        KSP_update(:,:,slice,:)=DD1;
    end
    clear KSP DD1 DD DD_phase newname D1
        end
    
    
    
    
    
    
    KSP2=KSP_update(:,:,:,1:end-1);  
    KSP_NOISE_update = KSP_update(:,:,:,end);
    clear KSP_update
    
    if 0
    % annoyinng noise_scale
    Q=sq(abs(KSP2(end-3,3:end-3,:,end)));  Q1=mean(Q(:));  % take image noise from the bottom of the volume, where the signal is lowest - JIC
    Q=sq(abs(KSP2(end-3,3,:,:)));  Q1=mean(Q(:));  % take image noise from the bottom of the volume, where the signal is lowest - JIC
   
    Q=sq(abs(KSP_NOISE(:,:,:,end)));  Q11=mean(Q(:));
    
    KSP_NOISE=KSP_NOISE/Q11*Q1;
    KSP_NOISE=KSP_NOISE(3:end-3,3:end-3,:);
    % KSP_NOISE=KSP_NOISE/Q1;
    end
    
    if ARG.DUAL<3  % noise might not be acquired correctly
        estimate_noise_from_data=0;
        
        if estimate_noise_from_data==1
            Q=sq((KSP2(end-3,2,:,:)));  Q1a=mean(abs(Q(:)));  % take image noise from the bottom of the volume, where the signal is lowest - JIC
            Q=sq((KSP2(end-3,end-2,:,:)));  Q1b=mean(abs(Q(:)));  Q1=min(Q1a,Q1b);
            
            Q=sq((KSP_NOISE(:,:,:,end)));  Q11=mean(abs(Q(:)));
            KSP_NOISE=KSP_NOISE/Q11*Q1;
        else
            KSP_NOISE=   KSP_NOISE /1;  %
            disp('change the scaling to the recon program');
        end
    end
    
     
    
    %w1=4; w2=4;  w3=4;
    %[w1 w2 w3]=kernel_size;
    w1=ARG.kernel_size(1);
    w2=ARG.kernel_size(2);
    w3=ARG.kernel_size(3);
    
    if ~isempty(ARG.truncate_length)
        KSP2=KSP2(:,:,:,1:ARG.truncate_length);
    end
    
    if 0 
    [w4]=ceil([ (w1*w2*w3*size(KSP2,4))^(1/3)]);
    stmp=size(KSP_NOISE);
    
    tmp=  KSP_NOISE(:); tmp=std(real(tmp));
    tmp=complex(randn(w1*w2*w3*size(KSP2,4),1), randn(w1*w2*w3*size(KSP2,4),1))*tmp;
    
    Noise=tmp(randperm(size(tmp,1),w1*w2*w3*size(KSP2,4)));
    end
    if 1
        Noise=KSP_NOISE;
        Noise=Noise(:);tmp=std(real(Noise));
        Noise=complex(randn(w1*w2*w3*size(KSP2,4),1), randn(w1*w2*w3*size(KSP2,4),1))*tmp;
        
        Noise=reshape(Noise(1:w1*w2*w3*size(KSP2,4)),[],size(KSP2,4));
        
        TT=Noise;
        [U,S,V]=svd(TT,'econ');  S=diag(S);
        [S(1)  S(2)]
        
    else
        
        TT=KSP2([5:end-5],3,:,:); %TT=KSP2([5:end-5],4,:,:);
        TT=reshape(TT,[],size(KSP2,4));        
        [~,SS,~]=svd(TT(1:w1*w2*w3,:),'econ'); SS=diag(SS); [SS(1) SS(2) ]
        
    end

    ARG.LLR_scale=max(abs(S)) ;
    
    for subrange=[10:5:size(TT,2)]
      [U,S1,V]=svd(TT(:,[1:subrange]),'econ');   S1=diag(S1);
       ARG.LLR_scale_subscale(subrange)=max(abs(S1)) ;
    end
    
    
    ARG.filename=filename;
    KSP_processed=zeros(1,size(KSP2,1)-w1+1);
    reading_file=1;
    writing_file=1; % if I save it I am master, so don't let anyone else be master
    
%   QQ.KSP2=KSP2;
   %QQ.KSP_recon=complex(single(zeros(size(KSP2))));  SHOULD BE IN MEMEORY  INSTEAD
%   QQ.ARG=ARG;   
if ARG.master_fast>0
    size_KSP=size(KSP2);
  save(filename_in,'size_KSP','ARG','writing_file','reading_file','KSP_processed','-v7.3');  % create nearly empty file

else
 save(filename_in,'KSP2','ARG','writing_file','reading_file','KSP_processed','-v7.3');  % create nearly empty file   
end

return



function [KSP2, meanphase, DD_phase]=sub_create_NVR_file_HCP(filename,ARG) 

tmp=1;
filename_in=([filename(1:end-4)     '_NVR.mat']);   % file with the data that will be used in the NVR, IE phase normalized
save(filename_in,'tmp');  % create nearly empty file

      
    newname=load(filename,'KSP');               
    matdim=size(newname.KSP(:,:,:,1:end-(ARG.DUAL-1)));  % just to indicate if the is noise + SB_refrence + gfactor  or noise +gfactor added
    
%    KSP=newname.KSP(:,:,:,1:end-(ARG.DUAL-1)); clear newname
KSP=newname.KSP;clear newname
    for nsl=size(KSP,3):-1:1
 %       MASK(:,:,nsl)= bwmorph( abs(KSP(:,:,nsl,1))>0,'shrink',1);
    end
    
    g_factor_map=mean(KSP(:,:,:,end)  ,4);
    KSP=KSP(:,:,:,1:end-1-(ARG.DUAL-1));  % removes the gfactor and the SB reference if it is there.
    
    for niter=1:size(KSP,4); KSP(:,:,:,niter)=KSP(:,:,:,niter)./(g_factor_map+eps); end
    
    
    
    KSP_NOISE = KSP(:,:,:,end);
   
  
  
  meanphase=1+0*mean(KSP(:,:,:,[1:end-1]),4);
  
    for slice=matdim(3):-1:1       
        for n=1:size(KSP,4)-1; % include the noise 
            KSP(:,:,slice,n)=KSP(:,:,slice,n).*exp(-i*angle(meanphase(:,:,slice)));
        end
    end
        DD_phase=0*KSP;
        
   for slice=matdim(3):-1:1   
      for n=1:size(KSP,4)-1; 
         tmp=KSP(:,:,slice,n);
            for ndim=[1:2]; tmp=ifftshift(ifft(ifftshift( tmp ,ndim),[],ndim),ndim+0); end
            [nx, ny, nc, nb] = size(tmp(:,:,:,:,1,1));
            tmp = bsxfun(@times,tmp,reshape(tukeywin(ny,1).^10,[1 ny]));
            tmp = bsxfun(@times,tmp,reshape(tukeywin(nx,1).^10,[nx 1]));
            for ndim=[1:2]; tmp=fftshift(fft(fftshift( tmp ,ndim),[],ndim),ndim+0); end
            DD_phase(:,:,slice,n)=tmp*0+1;   
      end
   end
      
    
     for slice=matdim(3):-1:1   
      for n=1:size(KSP,4); 
         KSP_update(:,:,slice,n)= KSP(:,:,slice,n).*exp(-i*angle( DD_phase(:,:,slice,n)   ));
      end
     end
     
        
      
    
    
    
    
    KSP2=KSP_update(:,:,:,1:end-1);  
    KSP_NOISE_update = KSP_update(:,:,:,end);
    clear KSP_update
        
    
     estimate_noise_from_data=1;
     
    if estimate_noise_from_data==1
    Q=sq((KSP2(end-3,2,:,:)));  Q1a=mean(abs(Q(:)));  % take image noise from the bottom of the volume, where the signal is lowest - JIC
    Q=sq((KSP2(end-3,end-2,:,:)));  Q1b=mean(abs(Q(:)));  Q1=min(Q1a,Q1b);
    
    Q=sq((KSP_NOISE(:,:,:,end)));  Q11=mean(abs(Q(:)));    
    KSP_NOISE=KSP_NOISE/Q11*Q1;
    else
     KSP_NOISE=   KSP_NOISE  *pi;  % 
     disp('change the scaling to the recon program');
    end
    
    
    
    
    %w1=4; w2=4;  w3=4;
    %[w1 w2 w3]=kernel_size;
    w1=ARG.kernel_size(1);
    w2=ARG.kernel_size(2);
    w3=ARG.kernel_size(3);
    
    if ~isempty(ARG.truncate_length)
        KSP2=KSP2(:,:,:,1:ARG.truncate_length);
    end
    
    if 0 
    [w4]=ceil([ (w1*w2*w3*size(KSP2,4))^(1/3)]);
    stmp=size(KSP_NOISE);
    
    tmp=  KSP_NOISE(:); tmp=std(real(tmp));
    tmp=complex(randn(w1*w2*w3*size(KSP2,4),1), randn(w1*w2*w3*size(KSP2,4),1))*tmp;
    
    Noise=tmp(randperm(size(tmp,1),w1*w2*w3*size(KSP2,4)));
    end
    if 1
        Noise=KSP_NOISE;
        Noise=Noise(:);tmp=std(real(Noise));
        Noise=complex(randn(w1*w2*w3*size(KSP2,4),1), randn(w1*w2*w3*size(KSP2,4),1))*tmp;
        
        Noise=reshape(Noise(1:w1*w2*w3*size(KSP2,4)),[],size(KSP2,4));
        
        TT=Noise;
        [U,S,V]=svd(TT,'econ'); S=diag(S);
        [S(1)  S(2)]
        
    else
        
        TT=KSP2([5:end-5],3,:,:); %TT=KSP2([5:end-5],4,:,:);
        TT=reshape(TT,[],size(KSP2,4));        
        [~,SS,~]=svd(TT(1:w1*w2*w3,:),'econ'); SS=diag(SS); [SS(1) SS(2) ]
        
    end

    ARG.LLR_scale=max(abs(S)) ;
    
    for subrange=[10:5:size(TT,2)]
      [U,S1,V]=svd(TT(:,[1:subrange]),'econ');   S1=diag(S1);
       ARG.LLR_scale_subscale(subrange)=max(abs(S1)) ;
    end
    
    
    ARG.filename=filename;
    KSP_processed=zeros(1,size(KSP2,1)-w1+1);
    reading_file=1;
    writing_file=1; % if I save it I am master, so don't let anyone else be master
    
if ARG.master_fast>0
    size_KSP=size(KSP2);
  save(filename_in,'size_KSP','ARG','writing_file','reading_file','KSP_processed','-v7.3');  % create nearly empty file

else
 save(filename_in,'KSP2','ARG','writing_file','reading_file','KSP_processed','-v7.3');  % create nearly empty file   
end

return



function [KSP2, meanphase, DD_phase]=sub_create_NVR_file_MPPCA(filename,ARG) 

tmp=1;
filename_in=([filename(1:end-4)     '_NVR.mat']);   % file with the data that will be used in the NVR, IE phase normalized
save(filename_in,'tmp');  % create nearly empty file

      
    newname=load(filename,'KSP');               
    matdim=size(newname.KSP(:,:,:,1:end-(ARG.DUAL-1)));  % just to indicate if the is noise + SB_refrence + gfactor  or noise +gfactor added
    
%    KSP=newname.KSP(:,:,:,1:end-(ARG.DUAL-1)); clear newname
KSP=abs(newname.KSP);clear newname  % make it magnitude
    for nsl=size(KSP,3):-1:1
 %       MASK(:,:,nsl)= bwmorph( abs(KSP(:,:,nsl,1))>0,'shrink',1);
    end
    
    g_factor_map=mean(KSP(:,:,:,end)  ,4)*0+1;  % no g-factor correction
    KSP=KSP(:,:,:,1:end-1-(ARG.DUAL-1));  % removes the gfactor and the SB reference if it is there.
    
    for niter=1:size(KSP,4); KSP(:,:,:,niter)=KSP(:,:,:,niter)./(g_factor_map+eps); end
    
    
    
    KSP_NOISE = KSP(:,:,:,end);
   
  
  
  meanphase=1+0*mean(KSP(:,:,:,[1:end-1]),4);
  
    for slice=matdim(3):-1:1       
        for n=1:size(KSP,4)-1; % include the noise 
            KSP(:,:,slice,n)=KSP(:,:,slice,n).*exp(-i*angle(meanphase(:,:,slice)));
        end
    end
        DD_phase=0*KSP;
        
   for slice=matdim(3):-1:1   
      for n=1:size(KSP,4)-1; 
         tmp=KSP(:,:,slice,n);
            for ndim=[1:2]; tmp=ifftshift(ifft(ifftshift( tmp ,ndim),[],ndim),ndim+0); end
            [nx, ny, nc, nb] = size(tmp(:,:,:,:,1,1));
            tmp = bsxfun(@times,tmp,reshape(tukeywin(ny,1).^10,[1 ny]));
            tmp = bsxfun(@times,tmp,reshape(tukeywin(nx,1).^10,[nx 1]));
            for ndim=[1:2]; tmp=fftshift(fft(fftshift( tmp ,ndim),[],ndim),ndim+0); end
            DD_phase(:,:,slice,n)=tmp*0+1;   
      end
   end
      
    
     for slice=matdim(3):-1:1   
      for n=1:size(KSP,4); 
         KSP_update(:,:,slice,n)= KSP(:,:,slice,n).*exp(-i*angle( DD_phase(:,:,slice,n)   ));
      end
     end
     
        
      
    
    
    
    
    KSP2=KSP_update(:,:,:,1:end-1);  
    KSP_NOISE_update = KSP_update(:,:,:,end);
    clear KSP_update
        
    
     estimate_noise_from_data=1;
     
    if estimate_noise_from_data==1
    Q=sq((KSP2(end-3,2,:,:)));  Q1a=mean(abs(Q(:)));  % take image noise from the bottom of the volume, where the signal is lowest - JIC
    Q=sq((KSP2(end-3,end-2,:,:)));  Q1b=mean(abs(Q(:)));  Q1=min(Q1a,Q1b);
    
    Q=sq((KSP_NOISE(:,:,:,end)));  Q11=mean(abs(Q(:)));    
    KSP_NOISE=KSP_NOISE/Q11*Q1;
    else
     KSP_NOISE=   KSP_NOISE  *pi;  % 
     disp('change the scaling to the recon program');
    end
    
    
    
    
    %w1=4; w2=4;  w3=4;
    %[w1 w2 w3]=kernel_size;
    w1=ARG.kernel_size(1);
    w2=ARG.kernel_size(2);
    w3=ARG.kernel_size(3);
    
    if ~isempty(ARG.truncate_length)
        KSP2=KSP2(:,:,:,1:ARG.truncate_length);
    end
    
    if 0 
    [w4]=ceil([ (w1*w2*w3*size(KSP2,4))^(1/3)]);
    stmp=size(KSP_NOISE);
    
    tmp=  KSP_NOISE(:); tmp=std(real(tmp));
    tmp=complex(randn(w1*w2*w3*size(KSP2,4),1), randn(w1*w2*w3*size(KSP2,4),1))*tmp;
    
    Noise=tmp(randperm(size(tmp,1),w1*w2*w3*size(KSP2,4)));
    end
    if 1
        Noise=KSP_NOISE;
        Noise=Noise(:);tmp=std(real(Noise));
        Noise=complex(randn(w1*w2*w3*size(KSP2,4),1), randn(w1*w2*w3*size(KSP2,4),1))*tmp;
        
        Noise=reshape(Noise(1:w1*w2*w3*size(KSP2,4)),[],size(KSP2,4));
        
        TT=Noise;
        [U,S,V]=svd(TT,'econ'); S=diag(S);  [S(1)  S(2)]
        
    else
        
        TT=KSP2([5:end-5],3,:,:); %TT=KSP2([5:end-5],4,:,:);
        TT=reshape(TT,[],size(KSP2,4));        
        [~,SS,~]=svd(TT(1:w1*w2*w3,:),'econ'); SS=diag(SS); [SS(1) SS(2) ]
        
    end

    ARG.LLR_scale=max(abs(S)) ;
    
    for subrange=[10:5:size(TT,2)]
      [U,S1,V]=svd(TT(:,[1:subrange]),'econ');   S1=diag(S1);
       ARG.LLR_scale_subscale(subrange)=max(abs(S1)) ;
    end
    
    
    ARG.filename=filename;
    KSP_processed=zeros(1,size(KSP2,1)-w1+1);
    reading_file=1;
    writing_file=1; % if I save it I am master, so don't let anyone else be master
    
if ARG.master_fast>0
    size_KSP=size(KSP2);
  save(filename_in,'size_KSP','ARG','writing_file','reading_file','KSP_processed','-v7.3');  % create nearly empty file

else
 save(filename_in,'KSP2','ARG','writing_file','reading_file','KSP_processed','-v7.3');  % create nearly empty file   
end

return


function [KSP2_tmp_update, KSP2_weight,NOISE,KSP2_tmp_update_threshold,energy_removed,SNR_weight]=subfunction_loop_for_NVR_avg_update(KSP2a,w3,w2,w1,lambda2,patch_avg, soft_thrs,KSP2_weight,ARG,NOISE,KSP2_tmp_update_threshold,energy_removed,SNR_weight)
  
  if ~isfield(ARG,'patch_scale'); patch_scale=1; else; patch_scale=ARG.patch_scale;end
  if ~exist('patch_avg'); patch_avg=1;end    %   patch_avg=0; means zero only
  if ~exist('soft_thrs'); soft_thrs=[]; end             
  
  if ~exist('KSP2_weight')
         KSP2_weight=zeros(size(KSP2a(:,:,:,1)));
  elseif isempty(KSP2_weight)
         KSP2_weight=zeros(size(KSP2a(:,:,:,1)));
  end
  
  if ~exist('NOISE_tmpKSP2_tmp_update')
        NOISE_tmp=zeros(size(KSP2a(:,:,:,1)));
  elseif isempty(KSP2_tmp_update)
        NOISE_tmp=zeros(size(KSP2a(:,:,:,1)));
  end
  
  if ~exist('KSP2_tmp_update_threshold')
         KSP2_tmp_update_threshold=zeros(size(KSP2a(:,:,:,1)));
  elseif isempty(KSP2_tmp_update_threshold)
         KSP2_tmp_update_threshold=zeros(size(KSP2a(:,:,:,1)));
  end
  
 if ~exist('energy_removed')
         energy_removed=zeros(size(KSP2a(:,:,:,1)));
  elseif isempty(energy_removed)
         energy_removed=zeros(size(KSP2a(:,:,:,1)));
 end
   
  
 if ~exist('SNR_weight')
         SNR_weight=zeros(size(KSP2a(:,:,:,1)));
  elseif isempty(SNR_weight)
        SNR_weight=zeros(size(KSP2a(:,:,:,1)));
  end
   
 
 
  KSP2_tmp_update=0*KSP2a;
  %NOISE=[];
  
          
                for n2=[1: max(1,floor(w2/ARG.patch_average_sub)):size(KSP2a,2)*1-w2+1  size(KSP2a,2)-w2+1];
                     for n3=[1: max(1,floor(w3/ARG.patch_average_sub)):size(KSP2a,3)*1-w3+1  size(KSP2a,3)-w3+1  ];
                  
                    KSP2_tmp=KSP2a(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:);
                        tmp1=reshape(KSP2_tmp,[],size(KSP2_tmp,4));
                        
                        [U,S,V]=svd([(tmp1) ],'econ'); S=diag(S);
                        
                        
                        
                        
                        [idx]=sum(S<lambda2);
                        if isempty(soft_thrs)
                            energy_scrub=sqrt(sum(S.^1)).\sqrt(sum(S(S<lambda2).^1));
                            S(S<lambda2)=0;
                            t=idx;
                          
                            
                        elseif soft_thrs==10  % USING MPPCA
                            
                            centering=0;
                            MM=size(tmp1,1);
                            NNN=size(tmp1,2);
                            R = min(MM, NNN);
                            scaling = (max(MM, NNN) - (0:R-centering-1)) / NNN;
                            scaling = scaling(:);
                            vals=S;
                            vals = (vals).^2 / NNN;
                            % First estimation of Sigma^2;  Eq 1 from ISMRM presentation
                            csum = cumsum(vals(R-centering:-1:1)); cmean = csum(R-centering:-1:1)./(R-centering:-1:1)'; sigmasq_1 = cmean./scaling;
                            % Second estimation of Sigma^2; Eq 2 from ISMRM presentation
                            gamma = (MM - (0:R-centering-1)) / NNN;
                            rangeMP = 4*sqrt(gamma(:));
                            rangeData = vals(1:R-centering) - vals(R-centering);
                            sigmasq_2 = rangeData./rangeMP;
                            t = find(sigmasq_2 < sigmasq_1, 1);
                             % NOISE(1:size(KSP2a,1),[1:w2]+(n2-1),[1:w3]+(n3-1),1) = sigmasq_2(t);
                            idx=size(S(t:end),1)  ;
                              energy_scrub=sqrt(sum(S.^1)).\sqrt(sum(S(t:end).^1)); 
                            S(t:end)=0;
                            
                            
                        else
                            S(max(1,end-floor(idx*soft_thrs)):end)=0;
                        end
                        
                        tmp1=U*diag(S)*V';
                        
                        tmp1=reshape(tmp1,size(KSP2_tmp));
                        
                        if patch_scale==1; else; patch_scale=size(S,1)-idx; end
                        
                        
                        if patch_avg==1
                            
                            KSP2_tmp_update(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:) =...
                                KSP2_tmp_update(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:) +patch_scale*tmp1;
                            KSP2_weight(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:) =...
                                KSP2_weight(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:) + patch_scale;
                            KSP2_tmp_update_threshold(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:) =...
                                KSP2_tmp_update_threshold(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:) +idx;
                            energy_removed(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:) =...
                                energy_removed(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:) +energy_scrub;
                            
                            SNR_weight(:,[1:w2]+(n2-1),[1:w3]+(n3-1),1) =...
                                SNR_weight(:,[1:w2]+(n2-1),[1:w3]+(n3-1),1) + S(1)./S(max(1,t-1));
                            
                            try
                                NOISE(1:size(KSP2a,1),[1:w2]+(n2-1),[1:w3]+(n3-1),1) = ...
                                    NOISE(1:size(KSP2a,1),[1:w2]+(n2-1),[1:w3]+(n3-1),1) +   sigmasq_2(t);; catch; end
                        else
                            KSP2_tmp_update(:,round(w2/2)+(n2-1),round(w3/2)+(n3-1),:) =...
                                KSP2_tmp_update(:,round(w2/2)+(n2-1),round(w3/2)+(n3-1),:) +patch_scale*tmp1(1,round(end/2),round(end/2),:);
                            KSP2_weight(:,round(w2/2)+(n2-1),round(w3/2)+(n3-1),:) =...
                                KSP2_weight(:,round(w2/2)+(n2-1),round(w3/2)+(n3-1),:) +patch_scale;
                            KSP2_tmp_update_threshold(:,round(w2/2)+(n2-1),round(w3/2)+(n3-1),:) =...
                                KSP2_tmp_update_threshold(:,round(w2/2)+(n2-1),round(w3/2)+(n3-1),:) +idx;
                            energy_removed(:,round(w2/2)+(n2-1),round(w3/2)+(n3-1),:)  =...
                                energy_removed(:,round(w2/2)+(n2-1),round(w3/2)+(n3-1),:) +energy_scrub;
                            
                            SNR_weight(:,round(w2/2)+(n2-1),round(w3/2)+(n3-1),1) =...
                                SNR_weight(:,round(w2/2)+(n2-1),round(w3/2)+(n3-1),1) + S(1)./S(max(1,t-1));
                            try
                                NOISE(:,round(w2/2)+(n2-1),round(w3/2)+(n3-1),:) = ...
                                    NOISE(:,round(w2/2)+(n2-1),round(w3/2)+(n3-1),:) +   sigmasq_2(t);; catch; end
                            
                        end
                        
                        
                     end
                end
                
            
             
             return

             



