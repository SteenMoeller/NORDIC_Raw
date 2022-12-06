function  create_NORDIC_data
sim_img_size=64;
sim_temp_size=48;
noise_level=0.01;
%%%%%%%%%%%%%%%   Create 4D array
Y=phantom3dAniso(sim_img_size);
val=unique(Y(:));

temp_basis=permute(dct(randn(6,size(val,1)),sim_temp_size),[2 1]);

for idx_val=1:size(val,1)
    MASK=(Y(:)==val(idx_val));
    IMG(MASK==1,1:size(temp_basis,2))=  repmat( temp_basis( idx_val,: ) ,[sum(MASK(:)) 1 ]);
end

IMG =reshape(IMG, [size(Y) sim_temp_size]); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  create spatially varying noise
gfactor_1D= hamming(sim_img_size);
gfactor_1D=gfactor_1D-min(gfactor_1D(:))+1;  % set it to be at least 1.
gfactor2D= gfactor_1D*gfactor_1D';
clear gfactor
for z=sim_img_size:-1:1
    gfactor(:,:,z)=gfactor2D*gfactor_1D(z);  % surely an easier way to expand the matrix
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


IID_NOISE=noise_level*complex(randn([sim_img_size sim_img_size sim_img_size sim_temp_size+1 ]),randn([sim_img_size sim_img_size sim_img_size sim_temp_size+1 ])   );

for t= sim_temp_size+1:-1:1
spatial_noise(:,:,:,t)= IID_NOISE(:,:,:,t).*gfactor;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add empty signal volume

IMG(:,:,:,end+1)=0;

%  Add noise

KSP=IMG+spatial_noise;

% add noise distribution


KSP(:,:,:,end+1)=gfactor;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% save output - legacy format 

KSP_processed=zeros(1,size(KSP,3));  % a counter in case things have to run in parallel

save(['demo_data_for_NORDIC'],'KSP','IMG','KSP_processed','-v7.3')


if 0
    script_for_creating_simulation_data
    NORDIC('demo_data_for_NORDIC.mat')
    
    QQ=load('KSP_demo_data_for_NORDICkernel8')
    Q=load('demo_data_for_NORDIC') 
    figure; clf
subplot(2,2,1); imagesc(squeeze(real(Q.KSP(:,:,32,12))),[0 1]); title('Data + noise')
subplot(2,2,2); imagesc(squeeze(real(Q.IMG(:,:,32,12))),[0 1]); title('Data w/o noise')
subplot(2,2,3); imagesc(squeeze(real(QQ.KSP_update(:,:,32,12))),[0 1]); title('NORDIC processed')
subplot(2,2,4); plot(squeeze(real(Q.KSP(20,25,32,1:end-2)  -   Q.IMG(20,25,32,1:end-1)))), hold on
                plot(squeeze(real(QQ.KSP_update(20,25,32,1:end)  -   Q.IMG(20,25,32,1:end-1))))
                legend('difference before NORDIC','difference after NORDIC')

 
end



return





