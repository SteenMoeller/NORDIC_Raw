function  NIFTI_NORDIC_dev3(fn_magn_in,fn_phase_in,fn_out,ARG2)
if 0  % program structure
    %  S1    Input_preparation
    %  S2    Data_load
    %  S3    Phase_preparation
    %  S4    g_factor_processing
    %  S5    NORDIC
    %  S6    Data_post_processing
    %  S7    Output_generation
end
% fMRI
%  fn_magn_in='name.nii.gz';
%  fn_phase_in='name2.nii.gz';
%  fn_out=['NORDIC_' fn_magn_in(1:end-7)];
%  ARG.temporal_phase=1;
%  ARG.phase_filter_width=10;
%  NIFTI_NORDIC(fn_magn_in,fn_phase_in,fn_out,ARG)
%
% dMRI
%  fn_magn_in='name.nii.gz';
%  fn_phase_in='name2.nii.gz';
%  fn_out=['NORDIC_' fn_magn_in(1:end-7)];
%  ARG.temporal_phase=3;
%  ARG.phase_filter_width=3;
%  NIFTI_NORDIC(fn_magn_in,fn_phase_in,fn_out,ARG)
%
%
%  file_input assumes 4D data
%
%OPTIONS
%   ARG.DIROUT    VAL=      string        Default is empty
%   ARG.noise_volume_last   VAL  = num  specifiec volume from the end of the series
%                                          0 default
%
%   ARG.factor_error        val  = num    >1 use higher noisefloor <1 use lower noisefloor
%                                          1 default
%
%   ARG.full_dynamic_range  val = [ 0 1]   0 keep the input scale, output maximizes range.
%                                            Default 0
%   ARG.temporal_phase      val = [1 2 3]  1 was default, 3 now in dMRI due tophase errors in some data
%   ARG.NORDIC              val = [0 1]    1 Default
%   ARG.MP                  val = [0 1 2]  1 NORDIC gfactor with MP estimation.
%                                          2 MP without gfactor correction
%                                          0 default
%   ARG.kernel_size_gfactor val = [val1 val2 val], defautl is [14 14 1]
%   ARG.kernel_size_PCA     val = [val1 val2 val], default is val1=val2=val3;
%                                                  ratio of 11:1 between spatial and temproal voxels
%   ARG.magnitude_only      val =[] or 1.  Using complex or magntiude only. Default is []
%                                          Function still needs two inputs but will ignore the second
%
%   ARG.save_add_info       val =[0 1];  If it is 1, then an additonal matlab file is being saved with degress removed etc.
%                                         default is 0
%   ARG.make_complex_nii    if the field exist, then the phase is being saved in a similar format as the input phase
%
%   ARG.phase_slice_average_for_kspace_centering     val = [0 1]
%                                         if val =0, not used, if val=1 the series average pr slice is first removed
%                                         default is now 0
%   ARG.phase_filter_width  val = [1... 10]  Specifiec the width of the smoothing filter for the phase
%                                         default is now 3
%
%   ARG.save_gfactor_map   val = [1 2].  1, saves the RELATIVE gfactor, 2 saves the
%                                            gfactor and does not complete the NORDIC processing
%   ARG.write_gzipped_niftis  val = [0, 1]  Output files are written to
%                                           disk ask compressed *.nii.gz NIfTIs.
%                                           default is 0 (= false).
%   ARG.phase_filter_name   val = 'tukey'  'DK'  'tukey' is the original implementation.
%                                           'DK' is decorrelated kernel from 10.1002/mrm.26138


%  VERSION September/2026
%  Copyright  Board of Regents, University of Minnesota, 2026
% [fList, pList] = matlab.codetools.requiredFilesAndProducts('NIFTI_NORDIC_dev2c.m');
%  S1    Input_preparation



ARG = default_options;  % Initialize options

f = fieldnames(ARG2);   % UPDATE INPUT FIELDS
for i = 1:length(f)
    ARG.(f{i}) = ARG2.(f{i});
end

ARG = update_options(ARG,fn_magn_in,fn_phase_in);

%  S2    Data_load

[II, info_phase, info,ARG]=DATA_load(fn_magn_in,fn_phase_in,ARG);

if size(II,4)<6
    disp('Too few volumes')
    return
end

%  S3    Phase_preparation

[KSP2,PHASES] = Phase_preparation(II,ARG);

%  S4    g_factor_processing
gfactor =g_factor_processing(KSP2,ARG);  % TO BE UPDATED
 
%  S5    NORDIC

[IMAGE , ARG ] =NORDIC_processing(KSP2,ARG,gfactor); % TO BE UPDATED


%  S6    Data_post_processing

[IMAGE, Residual] =Data_post_processing(IMAGE,gfactor, PHASES, KSP2, ARG);

%  S7    Output_generation

Output_generation(IMAGE, Residual, gfactor, info, info_phase, fn_out, ARG)

%%%%%%%%%%%%%%   END      %%%%%%%%%%%%%%%%



function  ARG = default_options;


ARG.DIROUT=[pwd '/'];
ARG.noise_volume_last=0;  % there is no noise volume   {0 1 2 ...} uses only a single volume
ARG.factor_error=1.0;  % error in gfactor estimatetion. >1 use higher noisefloor <1 use lower noisefloor
ARG.full_dynamic_range=0;  % saves the output on teh same scale as input data
ARG.temporal_phase=1;  % Correction for slice and time-specific phase
ARG.data_has_zero_elements=0; %  % If there are pixels that are constant zero
ARG.write_gzipped_niftis = 0;  % save the outputs as compressed files
ARG.phase_filter_width=3;  %  Uses a phase filter width with the tukey filter
ARG.NORDIC_patch_overlap=2;  % For the LLR how much overlap on full series
ARG.gfactor_patch_overlap=2;  % For the LLR how much overlap for gfactor estimation
ARG.kernel_size_PCA=[]; %  if empty sets it to the default
ARG.phase_slice_average_for_kspace_centering=1;
ARG.magnitude_only=0; %
ARG.save_gfactor_map=0; %
ARG.use_generic_NII_read=0;
ARG.NORDIC=1;  %  threshold based on Noise
ARG.MP=0;  % threshold based on Marchencko-Pastur
ARG.NVR_threshold=1;
ARG.patch_average=0;
ARG.phase_filter_name  =  'DK';
ARG.kernel_size_gfactor = [14 14 1 90];
%ARG.lambda2=0;
ARG.LLR_scale=1;
%ARG.calculate_residual=0;  % obsolete/redundant
ARG.save_residual_matlab=0;
ARG.save_residual_NIFTI=0;
ARG.make_complex_nii=0;  %  0 save magnitude only.   1 save magnitude and phase
ARG.save_add_info=0;  % saves out information in matlab for. Legacy compliance

ARG.use_magn_for_gfactor=0;
%ARG.soft_thrs_in=[];  % legacy flag

ARG.save_add_info_NOISE= 0;
ARG.save_add_info_Component_threshold = 0;
ARG.save_add_info_energy_removed = 0;
ARG.save_add_info_SNR_weight  = 0;

ARG.use_matlab_parpool=1;

ARG.interpolate_gfactor=1; % make the gfactor map smooth. Different from the original approach
ARG.NORDIC_wo_residuals=0;  % a scalar that determines if NORDIC shoudl be run with part of the normnal residuals removed

return


function  ARG = update_options(ARG,fn_magn_in,fn_phase_in);
if isempty(fn_phase_in);  ARG.magnitude_only=1;fn_phase_in=fn_magn_in; end
if ARG.use_generic_NII_read==1 ;  path(path,'/home/range6-raid1/moeller/matlab/ADD/NIFTI/'); end  % NO PYTHON EQUIVALENT
if ARG.MP==1 ;   ARG.NORDIC=0; ARG.soft_thrs=10; else ; ARG.soft_thrs=[]; end
if ARG.magnitude_only==1; ARG.temporal_phase=0; end
%if (ARG.save_residual_matlab==1  |  ARG.save_residual_NIFTI==1) & ARG.calculate_residual==0; ARG.calculate_residual=1; ;end


return





function Output_generation(IMG2, Residual, gfactor, info, info_phase, fn_out, ARG)


if ( ARG.save_gfactor_map==1 )
    info_gfactor=info;
    info_gfactor.ImageSize(4)=1;
    info_gfactor.Datatype='single';
    g_IMG=abs(gfactor(:,:,:,1:end)); % remove g-factor and noise for DUAL 1
    g_IMG(isnan(g_IMG))=0;
    g_IMG= single(abs(g_IMG));

    niftiwrite((g_IMG),[ARG.DIROUT 'gfactor_' fn_out(1:end) '.nii'], ...
        'Compressed', ARG.write_gzipped_niftis)
    restore_qform_from_source( ...
        [ARG.DIROUT 'gfactor_' fn_out(1:end) '.nii'], ...
        info_gfactor, ARG.write_gzipped_niftis);
end


if ARG.save_residual_matlab==1;
    %   Residual=KSP2-KSP_recon;
    save(fullfile(ARG.DIROUT, ['RESIDUAL' fn_out '.mat']),'Residual','-v7.3')
end

if ARG.make_complex_nii==1;
    IMG2_tmp=abs(IMG2(:,:,:,1:end)); % remove g-factor and noise for DUAL 1
    IMG2_tmp(isnan(IMG2_tmp))=0;
    tmp=sort(abs(IMG2_tmp(:)));  sn_scale=2*tmp(round(0.99*end));%sn_scale=max();
    gain_level=floor(log2(32000/sn_scale));
    %IMG2_tmp= int16(abs(IMG2_tmp)*2^gain_level);

    if  ARG.full_dynamic_range==0; gain_level=0;end

    if strmatch(info.Datatype,'uint16')
        IMG2_tmp= uint16(abs(IMG2_tmp)*2^gain_level);
    elseif strmatch(info.Datatype,'int16')
        IMG2_tmp= int16(abs(IMG2_tmp)*2^gain_level);
    else
        IMG2_tmp= single(abs(IMG2_tmp)*2^gain_level);
    end

    niftiwrite((IMG2_tmp),[ARG.DIROUT fn_out 'magn.nii'],info, ...
        'Compressed', ARG.write_gzipped_niftis)
    restore_qform_from_source( ...
        [ARG.DIROUT fn_out 'magn.nii'], ...
        info, ARG.write_gzipped_niftis);



    IMG2_tmp=angle(IMG2(:,:,:,1:end));
    if strmatch(info_phase.Datatype,'int16')
        %    IMG2_tmp=IMG2_tmp+pi;
    end

    IMG2_tmp=    (IMG2_tmp/(2*pi)+range_center)*range_norm;

    if strmatch(info_phase.Datatype,'uint16')
        IMG2_tmp= uint16(IMG2_tmp);
    elseif strmatch(info_phase.Datatype,'int16')
        IMG2_tmp= int16(IMG2_tmp);
    else
        IMG2_tmp= single((IMG2_tmp));
    end


    niftiwrite((IMG2_tmp),[ARG.DIROUT fn_out 'phase.nii'],info_phase, ...
        'Compressed', ARG.write_gzipped_niftis)
    restore_qform_from_source( ...
        [ARG.DIROUT fn_out 'phase.nii'], ...
        info_phase, ARG.write_gzipped_niftis);

else
    IMG2=abs(IMG2(:,:,:,1:end)); % remove g-factor and noise for DUAL 1
    IMG2(isnan(IMG2))=0;
    tmp=sort(abs(IMG2(:)));  sn_scale=2*tmp(round(0.99*end));%sn_scale=max();
    gain_level=floor(log2(32000/sn_scale));

    if  ARG.full_dynamic_range==0; gain_level=0;end

    if strmatch(info.Datatype,'uint16')
        IMG2= uint16(abs(IMG2)*2^gain_level);
    elseif strmatch(info.Datatype,'int16')
        IMG2= int16(abs(IMG2)*2^gain_level);
    elseif strmatch(info.Datatype,'double')
        IMG2= double(abs(IMG2)*2^gain_level);
    else
        IMG2= single(abs(IMG2)*2^gain_level);
    end
    if ARG.use_generic_NII_read==0;
        niftiwrite((IMG2),[ARG.DIROUT fn_out(1:end) '.nii'],info, ...
            'Compressed', ARG.write_gzipped_niftis)
        restore_qform_from_source( ...
            [ARG.DIROUT fn_out(1:end) '.nii'], ...
            info, ARG.write_gzipped_niftis);
    else
        nii=make_nii(IMG2);
        save_nii(nii, fullfile(ARG.DIROUT, [fn_out(1:end) '.nii']))
    end
end


if ARG.save_residual_NIFTI==1;
    %   Residual=KSP2-KSP_recon;
    Residual=abs(Residual(:,:,:,1:end)); % remove g-factor and noise for DUAL 1
    Residual(isnan(Residual))=0;
    %tmp=sort(abs(Residual(:)));  sn_scale=2*tmp(round(0.99*end));%sn_scale=max();
    %gain_level=floor(log2(32000/sn_scale));

    if  ARG.full_dynamic_range==0; gain_level=0;end

    if strmatch(info.Datatype,'uint16')
        Residual= uint16(abs(Residual)*2^gain_level);
    elseif strmatch(info.Datatype,'int16')
        Residual= int16(abs(Residual)*2^gain_level);
    elseif strmatch(info.Datatype,'double')
        Residual= double(abs(Residual)*2^gain_level);
    else
        Residual= single(abs(Residual)*2^gain_level);
    end

    niftiwrite((Residual),[ARG.DIROUT 'RESIDUAL' fn_out(1:end) '.nii'],info, ...
        'Compressed', ARG.write_gzipped_niftis)
    restore_qform_from_source( ...
        [ARG.DIROUT 'RESIDUAL' fn_out(1:end) '.nii'], ...
        info, ARG.write_gzipped_niftis);
end





if  ARG.save_add_info==1
    disp('saving additional info')
    save(fullfile(ARG.DIROUT, [fn_out '.mat']),'ARG2','ARG','-v7.3')
end




%  TODO



if ( ARG.save_add_info_NOISE==1 )
    info=info;
    info.ImageSize(4)=1;
    info.Datatype='single';
    IMG=abs(ARG.NOISE(:,:,:,1:end)); % remove g-factor and noise for DUAL 1
    IMG(isnan(IMG))=0;
    IMG= single(abs(IMG));

    niftiwrite((IMG),[ARG.DIROUT 'gfactor_post_normalization_' fn_out(1:end) '.nii'], ...
        'Compressed', ARG.write_gzipped_niftis)
    restore_qform_from_source( ...
        [ARG.DIROUT 'gfactor_post_normalization__' fn_out(1:end) '.nii'], ...
        info, ARG.write_gzipped_niftis);
end


if ( ARG.save_add_info_Component_threshold==1 )
    info=info;
    info.ImageSize(4)=1;
    info.Datatype='single';
    IMG=abs(ARG.Component_threshold(:,:,:,1:end)); % remove g-factor and noise for DUAL 1
    IMG(isnan(IMG))=0;
    IMG= single(abs(IMG));

    niftiwrite((IMG),[ARG.DIROUT 'Component_threshold_' fn_out(1:end) '.nii'], ...
        'Compressed', ARG.write_gzipped_niftis)
    restore_qform_from_source( ...
        [ARG.DIROUT 'Component_threshold_' fn_out(1:end) '.nii'], ...
        info, ARG.write_gzipped_niftis);
end


if ( ARG.save_add_info_energy_removed==1 )
    info=info;
    info.ImageSize(4)=1;
    info.Datatype='single';
    IMG=abs(ARG.energy_removed(:,:,:,1:end)); % remove g-factor and noise for DUAL 1
    IMG(isnan(IMG))=0;
    IMG= single(abs(IMG));

    niftiwrite((IMG),[ARG.DIROUT 'energy_removed_' fn_out(1:end) '.nii'], ...
        'Compressed', ARG.write_gzipped_niftis)
    restore_qform_from_source( ...
        [ARG.DIROUT 'energy_removed_' fn_out(1:end) '.nii'], ...
        info, ARG.write_gzipped_niftis);
end


if ( ARG.save_add_info_SNR_weight==1 )
    info=info;
    info.ImageSize(4)=1;
    inf.Datatype='single';
    IMG=abs(ARG.SNR_weight(:,:,:,1:end)); % remove g-factor and noise for DUAL 1
    IMG(isnan(IMG))=0;
    IMG= single(abs(IMG));

    niftiwrite((IMG),[ARG.DIROUT 'SNR_weight_' fn_out(1:end) '.nii'], ...
        'Compressed', ARG.write_gzipped_niftis)
    restore_qform_from_source( ...
        [ARG.DIROUT 'SNR_weight_' fn_out(1:end) '.nii'], ...
        info, ARG.write_gzipped_niftis);
end













return

function restore_qform_from_source(target_file, source_info, is_compressed)
%RESTORE_QFORM_FROM_SOURCE  Patch qform fields on a niftiwrite output.
%
%   MATLAB's niftiwrite populates sform_code from info.Transform.T but
%   leaves qform_code = 0, silently breaking downstream AFNI tools that
%   prefer qform over sform. This helper rewrites the qform_code and
%   qform parameters at the documented NIfTI-1 header byte offsets so
%   the output's qform matches SOURCE_INFO. The voxel data block is not
%   touched.
%
%   NIfTI-1 header offsets (https://nifti.nimh.nih.gov/nifti-1/):
%     pixdim[0]    offset  76    float32   (qfac)
%     qform_code   offset 252    int16
%     quatern_b    offset 256    float32
%     quatern_c    offset 260    float32
%     quatern_d    offset 264    float32
%     qoffset_x    offset 268    float32
%     qoffset_y    offset 272    float32
%     qoffset_z    offset 276    float32

if nargin < 3
    is_compressed = false;
end

if is_compressed
    gz_path = [target_file '.gz'];
    if ~exist(gz_path, 'file'); return; end
    gunzip(gz_path);
    delete(gz_path);
end

if ~exist(target_file, 'file'); return; end

fid = fopen(target_file, 'r+', 'l');
if fid == -1; return; end

fseek(fid, 76, 'bof');
fwrite(fid, single(source_info.raw.pixdim(1)), 'single');
fseek(fid, 252, 'bof');
fwrite(fid, int16(source_info.raw.qform_code), 'int16');
fseek(fid, 256, 'bof');
fwrite(fid, single(source_info.raw.quatern_b), 'single');
fwrite(fid, single(source_info.raw.quatern_c), 'single');
fwrite(fid, single(source_info.raw.quatern_d), 'single');
fwrite(fid, single(source_info.raw.qoffset_x), 'single');
fwrite(fid, single(source_info.raw.qoffset_y), 'single');
fwrite(fid, single(source_info.raw.qoffset_z), 'single');

fclose(fid);

if is_compressed
    gzip(target_file);
    delete(target_file);
end

return

function [IMG2, Residual] =Data_post_processing(IMG2,gfactor, PHASES, II, ARG);

Residual=ARG.Residual;
matdim = size(IMG2)    ;


for n=1:size(IMG2,4);
    IMG2(:,:,:,n)= IMG2(:,:,:,n).* gfactor;
end



for slice=matdim(3):-1:1
    for n=1:size(IMG2,4); % include the noise
        IMG2(:,:,slice,n)=IMG2(:,:,slice,n).*exp(i*angle(PHASES.meanphase(:,:,slice)));
    end
end

for slice=matdim(3):-1:1
    for n=1:size(IMG2,4);
        IMG2(:,:,slice,n)= IMG2(:,:,slice,n).*exp(i*angle( PHASES.DD_phase(:,:,slice,n)   ));
    end
end


for n=1:size(Residual,4);
    Residual(:,:,:,n)= Residual(:,:,:,n).* gfactor;
end



Residual = Residual.*ARG.ABSOLUTE_SCALE;
IMG2=IMG2.*ARG.ABSOLUTE_SCALE;

IMG2(isnan(IMG2))=0;
return

function gfactor =g_factor_processing(KSP2,ARG)

if isempty(ARG.kernel_size_gfactor) | size(ARG.kernel_size_gfactor,2)<3
    KSP2=(KSP2(:,:,1:end,1:min(90,end),1));  % should be at least 30 volumes
else
    KSP2=(KSP2(:,:,1:end,1:min(ARG.kernel_size_gfactor(4),end),1));
end


KSP2(isnan(KSP2))=0;
KSP2(isinf(KSP2))=0;

ARG.kernel_size=[ARG.kernel_size_gfactor(1) ARG.kernel_size_gfactor(2) 1];
ARG.kernel_size=[ARG.kernel_size_gfactor(1) ARG.kernel_size_gfactor(2) ARG.kernel_size_gfactor(3)];


ARG.patch_average_sub= ARG.gfactor_patch_overlap;
ARG.soft_thrs=10;  % MPPCa   (When Noise varies)

disp('estimating g-factor ...')
[KSP_recon,ARG,KSP_weight,NOISE,Component_threshold,energy_removed,SNR_weight] = updated_sub_LLR_Processing_v2(KSP2 ,ARG ) ;  % NEW


disp('completed estimating g-factor')
gfactor=sqrt(NOISE./KSP_weight);    %  NOISE./KSP_weight;

if sum(gfactor(:)==0)>0;  % gfactor stimation most likely failed since it is zero
    gfactor(isnan(gfactor))=0;
    gfactor(gfactor<1)=median(gfactor(gfactor~=0));
    ARG.data_has_zero_elements=1;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ARG.MP==2;
    gfactor=ones(size(gfactor));
end



if isempty(ARG.kernel_size_gfactor) | size(ARG.kernel_size_gfactor,2)<3
    KSP2=abs(KSP2(:,:,1:end,1:min(90,end),1));  % should be at least 30 volumes
else
    KSP2=abs(KSP2(:,:,1:end,1:min(ARG.kernel_size_gfactor(4),end),1));
end


if ARG.interpolate_gfactor==1
 blocksize=floor(ARG.kernel_size_gfactor([1 2])/ARG.gfactor_patch_overlap);
 for nslice=1:size(gfactor,3);
      gfactor(:,:,nslice) = (abs(   resample_2D(gfactor(1:blocksize(1):end,1:blocksize(2):end,nslice),size(gfactor(:,:,1)))  ));
 end
end




return

function  [KSP2,PHASES] = Phase_preparation(II,ARG)


KSP2=II;
matdim=size(KSP2);

tt=mean(reshape(abs(KSP2),[],size(KSP2,4)));
[idx]=find(tt>0.95*max(tt));
meanphase=mean(KSP2(:,:,:,idx(1)),4);

disp('estimating slice-dependent phases ...')
meanphase=mean(KSP2(:,:,:,[1:end-ARG.noise_volume_last]),4);
meanphase=meanphase*ARG.phase_slice_average_for_kspace_centering;

PHASES.meanphase= meanphase;

for slice=matdim(3):-1:1
    for n=1:size(KSP2,4); % include the noise
        KSP2(:,:,slice,n)=KSP2(:,:,slice,n).*exp(-i*angle(meanphase(:,:,slice)));
    end
end
DD_phase=0*KSP2;

if           ARG.temporal_phase>0; % Standarad low-pass filtered map

    switch ARG.phase_filter_name
        case 'tukey'
            tic;
            for slice=matdim(3):-1:1
                for n=1:size(KSP2,4);
                    tmp=KSP2(:,:,slice,n);
                    for ndim=[1:2]; tmp=ifftshift(ifft(ifftshift( tmp ,ndim),[],ndim),ndim+0); end
                    [nx, ny, nc, nb] = size(tmp(:,:,:,:,1,1));
                    tmp = bsxfun(@times,tmp,reshape(tukeywin(ny,1).^ARG.phase_filter_width,[1 ny]));
                    tmp = bsxfun(@times,tmp,reshape(tukeywin(nx,1).^ARG.phase_filter_width,[nx 1]));
                    for ndim=[1:2]; tmp=fftshift(fft(fftshift( tmp ,ndim),[],ndim),ndim+0); end
                    DD_phase(:,:,slice,n)=tmp;
                end
            end
            toc
        case 'DK'

            decorr_kernel = ([
                [0.03, 0.04, 0.05, 0.04, 0.03],
                [0.04, 0.08, 0.01, 0.08, 0.04],
                [0.05, 0.01, 0.00, 0.01, 0.05],
                [0.04, 0.08, 0.01, 0.08, 0.04],
                [0.03, 0.04, 0.05, 0.04, 0.03]
                ]) ;
            decorr_kernel = decorr_kernel / sum(  decorr_kernel(:) );

            if ARG.use_matlab_parpool==1;

                p = gcp("nocreate");
                if isempty(p); % if not running then create
                    parpool('Threads');  % delete(p)
                end

                PAR_RECON(matdim(3)) = parallel.FevalFuture;
                for slice=matdim(3):-1:1
                    PAR_RECON(slice)= parfeval(@sub_function_loop_dk,1,KSP2(:,:,slice,:), decorr_kernel );
                end

                for slice=matdim(3):-1:1
                    DD_phase(:,:,slice,:) = fetchOutputs( PAR_RECON(slice));
                end

            else
                for slice=matdim(3):-1:1
                    DD_phase(:,:,slice,:) = sub_function_loop_dk(KSP2(:,:,slice,:), decorr_kernel );
                end

            end


    end
end



if           ARG.temporal_phase==2; % Secondary step for filtered phase with residual spikes
    for slice=matdim(3):-1:1
        for n=1:size(KSP2,4);

            phase_diff=angle(KSP2(:,:,slice,n)./DD_phase(:,:,slice,n));
            mask=abs(phase_diff)>1;
            DD_phase2=DD_phase(:,:,slice,n);
            tmp=(KSP2(:,:,slice,n));
            DD_phase2(mask)= tmp(mask);% removes the phase 100%
            tmp2= abs(KSP2(:,:,slice,n)).*exp(1i*phase_diff);
            DD_phase2(mask) = tmp2(circshift(mask,[1 0])) + tmp2(circshift(mask,[0 1])) ...
                + tmp2(circshift(mask,[-1 0])) + tmp2(circshift(mask,[0 -1])) +tmp(mask);

            DD_phase2(mask) =       tmp(mask);
            DD_phase(:,:,slice,n)=DD_phase2; %
        end
    end
end


PHASES.DD_phase= DD_phase;


if           ARG.temporal_phase==3; % Secondary step for filtered phase with residual spikes

    for slice=matdim(3):-1:1
        for n=1:size(KSP2,4);
            phase_diff=angle(KSP2(:,:,slice,n)./DD_phase(:,:,slice,n));
            mask=abs(phase_diff)>1;
            DD_phase2=KSP2(:,:,slice,n);
            tmp2=(KSP2(:,:,slice,n));
            DD_phase2(mask) = tmp2(circshift(mask,[1 0])) + tmp2(circshift(mask,[0 1])) ...
                + tmp2(circshift(mask,[-1 0])) + tmp2(circshift(mask,[0 -1])) +0*tmp(mask);
            DD_phase3(:,:,slice,n)=DD_phase2; %
        end
    end
    KSP2 = abs(KSP2).*exp(1i*angle(DD_phase3));
    II = abs(II).*exp(1i*angle(DD_phase3)).*repmat(exp(1i*meanphase),[1 1 1 size(II,4)]);

    PHASES.DD_phase3= DD_phase3;  disp('NOT USED !!!')

end



for slice=matdim(3):-1:1
    for n=1:size(KSP2,4);
        KSP2(:,:,slice,n)= KSP2(:,:,slice,n).*exp(-i*angle( DD_phase(:,:,slice,n)   ));
    end
end




return

function  [II, info_phase, info,ARG] = DATA_load(fn_magn_in,fn_phase_in,ARG);

% remove some try statements

if ARG.magnitude_only~=1
    info_phase=niftiinfo(fn_phase_in);
    info=niftiinfo(fn_magn_in);


    I_M=abs(single(niftiread(fn_magn_in)));
    I_P=single(niftiread(fn_phase_in));

    phase_range=single(max(I_P(:)));
    phase_range_min=single(min(I_P(:)));

    info_phase.Datatype=class(I_P);
    info.Datatype=class(I_M);
    % Here, we combine magnitude and phase data into complex form
    fprintf('Phase should be -pi to pi...\n')

    % convert to single and then scale the phase
    I_P = single(I_P);
    range_norm=phase_range-phase_range_min;
    range_center=(phase_range+phase_range_min)/range_norm*1/2;
    I_P = (single(I_P)./range_norm -range_center)*2*pi;
    II=single(I_M)  .* exp(1i*I_P);



    fprintf('Phase data range is %.2f to %.2f\n', min(I_P(:)), max(I_P(:)))
else
    info=niftiinfo(fn_magn_in);
    I_M=abs(single(niftiread(fn_magn_in)));
    info.Datatype=class(I_M);
    info_phase=[];
    II=single(I_M);
end



TEMPVOL=abs(II(:,:,:,1));
ARG.ABSOLUTE_SCALE=min(TEMPVOL(TEMPVOL~=0));
II=II./ARG.ABSOLUTE_SCALE;







return

function   [IMG2, ARG]= NORDIC_processing(KSP2,ARG,gfactor);

ARG.matdim=size(KSP2);
ARG.kernel_size=repmat([ round((ARG.matdim(4)*11)^(1/3))   ],1,3);

for n=1:size(KSP2,4);
    KSP2(:,:,:,n)= KSP2(:,:,:,n)./gfactor;
end

if ARG.noise_volume_last>0
    KSP2_NOISE =KSP2(:,:,:,end+1-ARG.noise_volume_last);
else
    KSP2_NOISE=[];
end

if    ARG.data_has_zero_elements==1
    MASK=(sum(abs(KSP2),4)==0);
    Num_zero_elements=sum(MASK(:));
    for nvol=1:size(KSP2,4)
        tmp=KSP2(:,:,:,nvol);
        tmp(MASK)=(randn(Num_zero_elements,1)+1i*randn(Num_zero_elements,1))/sqrt(2);
        KSP2(:,:,:,nvol)=tmp;
    end
end

ARG=calculate_NORDIC_threshold(KSP2_NOISE, ARG);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('starting NORDIC ...')
[KSP_recon,ARG,KSP_weight,NOISE,Component_threshold,energy_removed,SNR_weight] = updated_sub_LLR_Processing_v2(KSP2 ,ARG ) ;  % NEW
KSP_recon=KSP_recon./repmat((KSP_weight),[1 1 1 size(KSP2,4)]);  % Assumes that the combination is with N instead of sqrt(N). Works for NVR not MPPCA
ARG.NOISE=  sqrt(NOISE./KSP_weight);
ARG.Component_threshold = Component_threshold./KSP_weight;
ARG.energy_removed = energy_removed./KSP_weight;
ARG.SNR_weight  = SNR_weight./KSP_weight;
IMG2=KSP_recon;
disp('completing NORDIC ...')


%if ARG.calculate_residual==1

ARG.Residual = KSP2 - KSP_recon;
if 0
    ARG.calculate_residual=2;  % means it will be calculated
    [Residual] = updated_sub_LLR_Processing_v2(KSP2 ,ARG ) ;  % NEW
    Residual=Residual./repmat((KSP_weight),[1 1 1 size(KSP2,4)]);  % Assumes that the combination is with N instead of sqrt(N). Works for NVR not MPPCA
    ARG.calculate_residual=1;
end



if ARG.NORDIC_wo_residuals~=0 % appears to make minor difference only
new_series= KSP2-ARG.NORDIC_wo_residuals*(ARG.Residual);
[KSP_recon,ARG,KSP_weight,NOISE,Component_threshold,energy_removed,SNR_weight] = updated_sub_LLR_Processing_v2( new_series ,ARG ) ;  % NEW
 KSP_recon=KSP_recon./repmat((KSP_weight),[1 1 1 size(KSP2,4)]);  % Assumes that the combination is with N instead of sqrt(N). Works for NVR not MPPCA
ARG.Residual = KSP2 - KSP_recon;
end





%end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

return


function  DD_phase_tmp= sub_function_loop_dk(KSP2,decorr_kernel)
slice=1;
for n=size(KSP2,4):-1:1;
    tmp=KSP2(:,:,slice,n);
    tmp_r = conv2( real(tmp) , decorr_kernel  ,'same');
    tmp_i = conv2( imag(tmp) , decorr_kernel  ,'same');
    tmp_ph= complex(tmp_r,tmp_i);
    DD_phase_tmp(:,:,1,n)=(tmp_ph);
end,

return


function [OUT,ARG,KSP_weight,NOISE,KSP2_tmp_update_threshold,energy_removed,SNR_weight] = updated_sub_LLR_Processing_v2(A ,ARG ) ;


w1=ARG.kernel_size(1); w2=ARG.kernel_size(2); w3=ARG.kernel_size(3);  % Kernel size
o1=0:floor(w1/ARG.patch_average_sub):(w1-1); o2=0:floor(w2/ARG.patch_average_sub):(w2-1); o3=0:floor(w3/ARG.patch_average_sub):(w3-1); % offset

if isempty(o3); o3=0; end

for q=0;
    for ostepx=o1
        %        for ostepy=o2
        %            for ostepz=o3

        s1= floor((size(A,1)-ostepx)/w1);
        % s2= floor((size(A,2)-ostepy)/w2);
        % s3= floor((size(A,3)-ostepz)/w3);    % number of patches does not have the offset
        for nstepx =[1:s1]
            REGIONS(q+1).RANGE=[nstepx ostepx ];
            REGIONS(q+1).RANGEy = o2;
            REGIONS(q+1).RANGEz = o3;  q=q+1;  % blocks that are used
        end
        range_used=[1:w1]+(nstepx-1)*w1+ostepx;
        if   range_used(end)<size(A,1)
            edgewidth=size(A,1)-range_used(end);
            REGIONS(q+1).RANGE=[nstepx+1 edgewidth-w1];
            REGIONS(q+1).RANGEy = o2;
            REGIONS(q+1).RANGEz = o3;  q=q+1;  % blocks that are used
        end
        %            end
        %        end
    end
end





[OUT,AUX_OUT]= initialize_arrays(A);

clear PAR_RECON  ARG_IN  AUX
PAR_RECON(size(REGIONS,2)) = parallel.FevalFuture;
ARG_IN.kernel_size=ARG.kernel_size;
ARG_IN.Component_threshold_to_use=[];  % needs update
ARG_IN.soft_thrs=ARG.soft_thrs;
%ARG_IN.lambda2=ARG.lambda2;
ARG_IN.NVR_threshold=ARG.NVR_threshold;
%ARG_IN.calculate_residual=ARG.calculate_residual;

% regions_step=23; ARG_IN.RANGE=REGIONS(regions_step).RANGE;[AA,AUX, ARG_IN] =   process_subset(A,ARG_IN );
if ARG.use_matlab_parpool==1;
    for regions_step =[1:size(REGIONS,2)]
        ARG_IN.RANGE =REGIONS(regions_step).RANGE;
        ARG_IN.RANGEy=REGIONS(regions_step).RANGEy;
        ARG_IN.RANGEz=REGIONS(regions_step).RANGEz;
        PAR_RECON(regions_step)= parfeval(@process_subset,3, A,ARG_IN );  % sending all of A instead of just subset
    end
end

for nstepx = [1:size(REGIONS,2)]
    % [AA,AUX] = fetchOutputs( PAR_RECON(nstepx));
    %[AA,AUX, ARG_IN] = fetchOutputs( PAR_RECON(nstepx));  % ARG_in is redundant to carry through
    if ARG.use_matlab_parpool==1;
       % [idx,AA,AUX, ARG_IN] = fetchNext( PAR_RECON(nstepx));  % ARG_in is redundant to carry through
        [AA,AUX, ARG_IN] = fetchOutputs( PAR_RECON(nstepx));  % ARG_in is redundant to carry through
    else
        ARG_IN.RANGE =REGIONS(regions_step).RANGE;
        ARG_IN.RANGEy=REGIONS(regions_step).RANGEy;
        ARG_IN.RANGEz=REGIONS(regions_step).RANGEz;
        [AA,AUX, ARG_IN] = process_subset(A,ARG_IN );  % sending all of A instead of just subset
    end

    w1=ARG_IN.kernel_size(1);
    nstepx= ARG_IN.RANGE(1);
    ostepx= ARG_IN.RANGE(2);

    OUT([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  = OUT([1:w1]+(nstepx-1)*w1+ostepx,:,:,:) + AA;

    AUX_OUT.KSP2_weight([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  = AUX_OUT.KSP2_weight([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  +  AUX.KSP2_weight;
    AUX_OUT.KSP2_tmp_update_threshold([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  = AUX_OUT.KSP2_tmp_update_threshold([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  +  AUX.KSP2_tmp_update_threshold;
    AUX_OUT.energy_removed([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  = AUX_OUT.energy_removed([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  +  AUX.energy_removed;
    AUX_OUT.SNR_weight([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  = AUX_OUT.SNR_weight([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  +  AUX.SNR_weight;
    AUX_OUT.NOISE([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  = AUX_OUT.NOISE([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  +  AUX.NOISE;


    %  [OUT,AUX_OUT]= subfunction_update_matrix_global(OUT,AUX_OUT,AA,AUX,ARG_IN);
    % A(:,:,:,[1:w1]+(nstepx-1)*w1,:,:,:)=AA;
end


KSP_weight=AUX_OUT.KSP2_weight;
NOISE=AUX_OUT.NOISE;
KSP2_tmp_update_threshold=AUX_OUT.KSP2_tmp_update_threshold;
energy_removed=AUX_OUT.energy_removed;
SNR_weight=AUX_OUT.SNR_weight;



return


function [OUT,AUX]= initialize_arrays(IN);

OUT =zeros(size(IN(:,:,:,:)));  % Array that will hold updated image
AUX.KSP2_weight=zeros(size(IN(:,:,:,1)));  % Array that will hold a counter for how many times a voxel has been used
AUX.KSP2_tmp_update_threshold=zeros(size(IN(:,:,:,1)));  % Array that will hold how many singular values retained
AUX.energy_removed=zeros(size(IN(:,:,:,1)));  % Array that will hold the l2 energy of the noise
AUX.SNR_weight=zeros(size(IN(:,:,:,1)));  % Array that will hold the ratio between highest and lowest retained singular value
AUX.NOISE=zeros(size(IN(:,:,:,1)));  % Array that will hold the lowest estimated signal singular value from MPPCA


return

function [OUT,AUX_OUT]= subfunction_update_matrix_global(OUT,AUX_OUT,AA,AUX,ARG_IN);
% update parsed out sections of data

w1=ARG_IN.kernel_size(1);
nstepx= ARG_IN.RANGE(1);
ostepx= ARG_IN.RANGE(2);


OUT([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  = OUT([1:w1]+(nstepx-1)*w1+ostepx,:,:,:) + AA;

AUX_OUT.KSP2_weight([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  = AUX_OUT.KSP2_weight([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  +  AUX.KSP2_weight;
AUX_OUT.KSP2_tmp_update_threshold([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  = AUX_OUT.KSP2_tmp_update_threshold([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  +  AUX.KSP2_tmp_update_threshold;
AUX_OUT.energy_removed([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  = AUX_OUT.energy_removed([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  +  AUX.energy_removed;
AUX_OUT.SNR_weight([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  = AUX_OUT.SNR_weight([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  +  AUX.SNR_weight;
AUX_OUT.NOISE([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  = AUX_OUT.NOISE([1:w1]+(nstepx-1)*w1+ostepx,:,:,:)  +  AUX.NOISE;

return


function  ARG=calculate_NORDIC_threshold(KSP2_NOISE, ARG);


if ARG.noise_volume_last>0
    %tmp_noise=KSP2(:,:,:,end+1-ARG.noise_volume_last);
    tmp_noise=KSP2_NOISE;

    tmp_noise(isnan(tmp_noise))=0;
    tmp_noise(isinf(tmp_noise))=0;
    ARG.measured_noise=std(tmp_noise(tmp_noise~=0));  % sqrt(2) for real and complex
else
    ARG.measured_noise=1;  % IF COMPLEX DATA
end


if  ARG.use_magn_for_gfactor==0 & (isempty(ARG.magnitude_only) | ARG.magnitude_only==0)  %% WOULD THIS BE THE ISSUE  & replaced by |
    ARG.measured_noise =  ARG.measured_noise/sqrt(2);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if  ~isempty(ARG.kernel_size_PCA)
    ARG.kernel_size = ARG.kernel_size_PCA ;
end


if ARG.matdim(3) <= ARG.kernel_size(3)  % Number of slices is less than cubic kernel
    ARG.kernel_size   = repmat([ round((ARG.matdim(4)*11/matdim(3) )^(1/2))   ],1,2);
    ARG.kernel_size(3)= matdim(3);
end


QQ.KSP_processed=zeros(1,ARG.matdim(1)-ARG.kernel_size(1));
ARG.patch_average=0;
ARG.patch_average_sub= ARG.NORDIC_patch_overlap ;
% ARG.kernel_size=[7 7 7]; ARG.patch_average_sub=7;  MPPCA
% ARG.soft_thrs=10;  % MPPCa   (When Noise varies)
ARG.LLR_scale=1;
ARG.NVR_threshold=0;

%ARG.soft_thrs=[];  % NORDIC  (When noise is flat)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ARG.NVR_threshold=0;
for ntmp=1:10
    [~,S,~]=svd(randn(prod(ARG.kernel_size),ARG.matdim(4) ))  ;
    ARG.NVR_threshold=ARG.NVR_threshold+S(1,1);
end

if ARG.magnitude_only~=1  % 4/29/2021
    ARG.NVR_threshold= ARG.NVR_threshold/10*sqrt(2)* ARG.measured_noise*ARG.factor_error;  % sqrt(2) due to complex  1.20 due to understimate of g-factor
else
    ARG.NVR_threshold= ARG.NVR_threshold/10*ARG.measured_noise*ARG.factor_error;  % sqrt(2) due to complex  1.20 due to understimate of g-factor
end

%if ARG.MP>0
%    ARG.soft_thrs=10;
%end


%if ~isempty(ARG.soft_thrs_in)
%    ARG.soft_thrs=ARG.soft_thrs_in;
%end


return

function  [OUT, OUT_ARRAY, ARG] = process_subset(IN,ARG)

%  ARG.Component_threshold_to_use used for prunning instead of hard
%  threshold



w1=ARG.kernel_size(1);
w2=ARG.kernel_size(2);
w3=ARG.kernel_size(3);  % Kernel size

nstepx= ARG.RANGE(1);
ostepx= ARG.RANGE(2);
%ostepy= ARG.RANGE(3);
%ostepz= ARG.RANGE(4);

s1= floor((size(IN,1)-ostepx)/w1);
%s2= floor((size(IN,2)-ostepy)/w2);
%s3= floor((size(IN,3)-ostepz)/w3);

% AA=IN(:,:,:,[1:w1]+(nstepx-1)*w1+ostepx,:,:,:); % reduce to work on
% subset can be faster memory access
AA=IN([1:w1]+(nstepx-1)*w1+ostepx,:,:,:); % reduce to work on subset

[OUT, OUT_ARRAY] = calc_subset(AA, ARG );



return

function   [OUT,OUT_ARRAY] = calc_subset(IN,ARG)
%IN=permute(IN,[4 1 2 3]);  % input structure was shifted for memeory efficiency. Shifting back to "normal" format

Component_threshold_to_use=ARG.Component_threshold_to_use;
lambda2=ARG.NVR_threshold;
soft_thrs=ARG.soft_thrs;

%w1=ARG.kernel_size(1);
w2=ARG.kernel_size(2);
w3=ARG.kernel_size(3);  % Kernel size

% nstepx= ARG.RANGE(1);
% ostepx= ARG.RANGE(2);
%ostepy= ARG.RANGEy;
%ostepz= ARG.RANGEz;

% s1= floor((size(IN,1)-ostepx)/w1);
%s2= floor((size(IN,2)-ostepy)/w2);
%s3= floor((size(IN,3)-ostepz)/w3);

%initialize output fields

IN_UPDATE =zeros(size(IN(:,:,:,:)));  % Array that will hold updated image
KSP2_weight=zeros(size(IN(:,:,:,1)));  % Array that will hold a counter for how many times a voxel has been used
KSP2_tmp_update_threshold=zeros(size(IN(:,:,:,1)));  % Array that will hold how many singular values retained
energy_removed=zeros(size(IN(:,:,:,1)));  % Array that will hold the l2 energy of the noise
SNR_weight=zeros(size(IN(:,:,:,1)));  % Array that will hold the ratio between highest and lowest retained singular value
NOISE=zeros(size(IN(:,:,:,1)));  % Array that will hold the lowest estimated signal singular value from MPPCA

for ostepy= ARG.RANGEy;
    for ostepz= ARG.RANGEz;
        s2= floor((size(IN,2)-ostepy)/w2);
        s3= floor((size(IN,3)-ostepz)/w3);
        for n2=[(0:s2-1)*w2+1+ostepy  size(IN,2)-w2+1 ] % determined steps to extract data from, index offset to start at 1, offset by overlap
            for n3= [(0:s3-1)*w3+1+ostepz  size(IN,3)-w3+1]  % determined steps to extract data from, index offset to start at 1, offset by overlap

                % for n2=[1: max(1,floor(w2/ARG.patch_average_sub)):size(KSP2a,2)*1-w2+1  size(KSP2a,2)-w2+1];
                %      for n3=[1: max(1,floor(w3/ARG.patch_average_sub)):size(KSP2a,3)*1-w3+1  size(KSP2a,3)-w3+1  ];

                if ~isempty(Component_threshold_to_use)
                    Component_threshold_to_use_tmp=Component_threshold_to_use(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:);
                end
                if exist('Component_threshold_to_use_tmp')
                    Component_threshold_to_use_tmp = min(Component_threshold_to_use_tmp(:));
                else
                    Component_threshold_to_use_tmp=[];  % initialize if not defined
                end

                [update, idx_scalar, energy_scrub_scalar , SNR_weight_scalar, NOISE_scalar]= ...
                    LLR_NORDIC_MPPCA(IN(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:),lambda2,soft_thrs,Component_threshold_to_use_tmp,ARG);

                [IN_UPDATE,KSP2_weight,KSP2_tmp_update_threshold,energy_removed,SNR_weight,NOISE]= ...
                    subfunction_update_matrix_local(update, idx_scalar, energy_scrub_scalar , SNR_weight_scalar, NOISE_scalar,w2, n2, w3, n3,...
                    IN_UPDATE,KSP2_weight,KSP2_tmp_update_threshold,energy_removed,SNR_weight,NOISE);

            end
        end
    end
end






OUT=IN_UPDATE;

OUT_ARRAY.KSP2_weight=KSP2_weight;
OUT_ARRAY.KSP2_tmp_update_threshold=KSP2_tmp_update_threshold;
OUT_ARRAY.energy_removed=energy_removed;
OUT_ARRAY.SNR_weight=SNR_weight;
OUT_ARRAY.NOISE=NOISE;

return

function [IN_UPDATE,KSP2_weight,KSP2_tmp_update_threshold,energy_removed,SNR_weight,NOISE]= subfunction_update_matrix_local(update, idx_scalar, energy_scrub_scalar , SNR_weight_scalar, NOISE_scalar,w2, n2, w3, n3,IN_UPDATE,KSP2_weight,KSP2_tmp_update_threshold,energy_removed,SNR_weight,NOISE);


IN_UPDATE(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:)=IN_UPDATE(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:) + update;
KSP2_weight(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:)  =KSP2_weight(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:)  + 1;
KSP2_tmp_update_threshold(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:)  =KSP2_tmp_update_threshold(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:)  + idx_scalar;
energy_removed(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:)  =energy_removed(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:)  + energy_scrub_scalar;
SNR_weight(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:)  =SNR_weight(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:)  + SNR_weight_scalar;
NOISE(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:)  =NOISE(:,[1:w2]+(n2-1),[1:w3]+(n3-1),:)  +NOISE_scalar;

return

function  [tmp1, idx, energy_scrub , SNR_weight, NOISE]=LLR_NORDIC_MPPCA(IN,lambda2,soft_thrs,Component_threshold_to_use_tmp,ARG);

tmp1=reshape(IN,[],size(IN,4));

[U,S,V]=svd([(tmp1) ],'econ'); S=diag(S);

if ~isempty(Component_threshold_to_use_tmp)
    lambda2 =S(Component_threshold_to_use_tmp)%; % Or should it be +1 or are the those to keep
end



% lambda2=min(S(2),lambda2);

if isempty(soft_thrs)   % NORDIC threshold
    [idx]=sum(S<lambda2);
    energy_scrub=sqrt(sum(S.^1)).\sqrt(sum(S(S<lambda2).^1));
    S(S<lambda2)=0;
    t=idx;
elseif soft_thrs==10  % USING MPPCA threshold
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
    %sigmasq_1 = sigmasq_1*sqrt(5/8);
    % Second estimation of Sigma^2; Eq 2 from ISMRM presentation
    gamma = (MM - (0:R-centering-1)) / NNN;
    rangeMP = 4*sqrt(gamma(:));
    rangeData = vals(1:R-centering) - vals(R-centering);
    sigmasq_2 = rangeData./rangeMP;
    t = find(sigmasq_2 < sigmasq_1, 1);
    % NOISE(1:size(KSP2a,1),[1:w2]+(n2-1),[1:w3]+(n3-1),1) = sigmasq_2(t);
    idx=size(S(t:end),1)  ;
    energy_scrub=sqrt(sum(S.^1)).\sqrt(sum(S(t:end).^1));
    S(t:end)=0; % always keep at least one
else
    [idx]=sum(S<lambda2);
    S(max(1,end-floor(idx*soft_thrs)):end)=0;
end

tmp1=U*diag(S)*V';
tmp1=reshape(tmp1,size(IN));
if exist('sigmasq_2'); NOISE = sigmasq_2(t); else ; NOISE =0; end
SNR_weight=S(1)./S(max(1,t-1));

%if ARG.calculate_residual==2
%    tmp1=tmp1-IN;
%end

return


function maps2aa = resample_2D(maps2aa_in,ssize);


in_size=size(maps2aa_in);
if size(in_size,2)<4; in_size(4)=1; end
if in_size(3)==0; in_size(3)=1; end

maps2aa = maps2aa_in;
maps2aa = FFT_MR(maps2aa,[1  2],1);
ww=0.3;
tmp=tukeywin(in_size(1),ww)*tukeywin(in_size(2),ww)';
for nch=1:in_size(3); maps2aa(:,:,nch)= maps2aa(:,:,nch) .* tmp; end
maps2aa = FFT_MR(maps2aa,[1  2],-1);
%maps2aa = maps2aa_in;
if size(maps2aa,2)~=ssize(2)
maps2aa = FFT_MR(maps2aa,2,1);
maps2aa(1,ssize(2),1,1)=0;
maps2aa = circshift(   maps2aa,[0  round((ssize(2)-in_size(2)) /2 ) 0] );
maps2aa = FFT_MR(maps2aa,2,-1);
end

if size(maps2aa,1)~=ssize(1)
maps2aa = FFT_MR(maps2aa,1,1);
maps2aa(ssize(1),1,1,1)=0;
maps2aa = circshift(   maps2aa,[ round((ssize(1)-in_size(1)) /2 ) 0] );
maps2aa = FFT_MR(maps2aa,1,-1);
end

try
    if size(maps2aa,4)~=ssize(4) & size(maps2aa,4)>1
        maps2aa = FFT_MR(maps2aa,4,1);
        maps2aa(1,1,1,ssize(4))=0;
        maps2aa = FFT_MR(maps2aa,4,-1);
    end
catch ; end

maps2aa_phase = maps2aa;

maps2aa = abs(maps2aa_in);

for nsl=in_size(4):-1:1
    for nch=in_size(3):-1:1
        maps2aa_magn(:,:,nch,nsl)=interp2( abs(maps2aa(:,:,nch,nsl)),linspace(1,in_size(2),ssize(2)), linspace(1,in_size(1),ssize(1))' ,'cubic' );
    end
end

try
    if size(maps2aa_magn,4)~=ssize(4) & size(maps2aa_magn,4)>1
        maps2aa_magn = FFT_MR(maps2aa_magn,4,1);
        maps2aa_magn(1,1,1,ssize(4))=0;
        maps2aa_magn = FFT_MR(maps2aa_magn,4,-1);
    end
catch ; end

maps2aa = maps2aa_magn  .*exp(1i*angle(maps2aa_phase));
return

function  MR=FFT_MR(MR,ndim,version)
fft_MR = @(x,ndim)    fftshift(fft(fftshift(x  ,ndim  ),[],ndim),ndim)/sqrt(size(x,ndim)) ;   %
ifft_MR = @(x,ndim)    ifftshift(ifft(ifftshift(x  ,ndim  ),[],ndim),ndim)*sqrt(size(x,ndim)) ;  %  noise-scaling

if version==1
  for nndim=ndim;  MR = fft_MR(MR,nndim); end
elseif version==-1
 for nndim=ndim;  MR = ifft_MR(MR,nndim); end
end
return









            
