rootpath = '//path/to/repo//';
impath = strcat(rootpath, 'data/glass_movement_video/');
resultspath = strcat(rootpath, 'data/glass_movement_results/');

%% segments the filaments in each frame
% basic segmentation is v easy
% most of the code is then trying to 'cut up' overlapping/crossing filaments
% we save the output from each frame as a .mat variable containing lots of info

tic
for frame = 1:499
im = imread( strcat(impath,sprintf('Fused_Red_24-02-12_080-083%04d.tif',frame)),'tif' );
% binarise and skeletonise
binary = imbinarize(im);
bw2 = bwmorph(binary,'thin',Inf);
bw2 = bwmorph(bw2,'fill');
bw3 = bwskel(bw2,'minbranchlength',5);
bw3 = bwmorph(bw3,'spur');

s = regionprops(bw3,'PixelIdxList','PixelList','Centroid');
segmentcount = 0;
var = length(s);
% segment filaments that have multiple end points
filter = true(length(s),1);
for i = 1:length(s)
    s(i).Length = length(s(i).PixelIdxList);
    filament = zeros(size(bw3),'int8');
    filament(s(i).PixelIdxList) = 1;    
    % segment filaments with > 2 endpoints
    n_endpoints = sum(bwmorph(filament,'endpoints'),'all');
    if n_endpoints>2   
        segmentcount = segmentcount+1;
        temp = (bwmorph(filament,'branchpoints'));
        [y,x] = find(temp);
        y = mean(y,1);
        x = mean(x,1);        
        temp = double(s(i).PixelList)-[x,y];
        r = (temp(:,1).^2 + temp(:,2).^2).^0.5;      
        angles = atan2(  temp(:,2),temp(:,1)  );
        % figure()
        % histogram(angles(r< 30),20)
        [counts_, edges] = histcounts(angles(r< 15),-pi:(pi/10):pi);
        bincentres = edges + pi/20;
        bincentres = [bincentres bincentres(end)+pi/20];
        counts = [counts_ counts_(1:2)]; 
        % figure();
        % plot(bincentres,counts)
        [~,locs] = findpeaks(counts, bincentres,'SortStr','descend');
        if n_endpoints == 3
         try
            locs = locs(1:3);
        catch
            filter(i) = false;
            continue
        end
        gaps = [0 0 0];
        for j = 1:3
            this = locs;
            this(j) = [];
            gaps(j) = abs(this(2)- this(1));
        end
        [~,idx] = min(abs(gaps -pi));
        joiningangle = locs(idx); % rads
        newlist = s(i).PixelIdxList;
        angles_thresh = abs(angles-joiningangle)< 1; 
        % anything close to branch point and with angles close to joinangle
        % is removed
        newlist(r<5 & angles_thresh) = []; 
        %newlist = newlist(r > 3 | angles_thresh);
        
        test = zeros(size(bw3),'int8');
        test(newlist) = 1; 
        s2 = regionprops(imbinarize(test),'PixelIdxList','PixelList','Centroid');
        if length(s2) >2
             test = imdilate(test,ones(3));
             test = bwmorph(test,'thin',Inf);
            s2 = regionprops(test,'PixelIdxList','PixelList','Centroid');
        end
        if length(s2) == 2       
            s2(1).Length = length(s2(1).PixelIdxList);
            s(i) = s2(1);
            newidx = length(s)+1;
            s2(2).Length = length(s2(2).PixelIdxList);
            s(newidx) = s2(2);            
            % figure();
            % hold on
            % imagesc(test)
            % title(i)          
            % scatter(s(i).Centroid(1),s(i).Centroid(2),'filled');
            % scatter(s2(2).Centroid(1),s2(2).Centroid(2),'filled')
        end

        end 
        if n_endpoints == 4
            try
                locs = locs(1:4);
            catch
                filter(i) = false;
                continue
            end
            locs = sort(locs);
            % first filament
            newlist = s(i).PixelIdxList;
            angles_thresh = (abs(angles-locs(1))< 0.7 | abs(angles-locs(3))< 0.7); 
            newlist(r > 2 & r<10 & ~angles_thresh) = []; 
            test = zeros(size(bw3),'int8');
            test(newlist) = 1;    
   
            s2 = regionprops(imbinarize(test),'PixelIdxList','PixelList','Centroid');
            if length(s2) == 3
                minvals = [0 0 0];
                for j = 1:3
                    what = s2(j).PixelList - [x y];
                    what = what(:,1).^2 + what(:,2).^2;
                    minvals(j) = min(what);
                end
                [~,idx] = min(minvals);
                filament1 = s2(idx);
                repair = zeros(size(test),'int8');
                repair(s(i).PixelIdxList) = 1;
                repair(s2(idx).PixelIdxList) = 0;
                repair = imdilate(repair,ones(22));
                repair = bwmorph(repair,'thin',Inf);
                s3 = regionprops(repair,'PixelIdxList','PixelList','Centroid'); 

                filament1.Length = length(filament1.PixelIdxList);
                s(i) = filament1;
               
                newidx = length(s)+1;
                s3(1).Length = length(s3(1).PixelIdxList);
                s(newidx) = s3(1);

            % figure(); hold on
            % imagesc(test)
            % scatter(s3.Centroid(1),s3.Centroid(2),'filled');
            % scatter(filament1.Centroid(1),filament1.Centroid(2),'filled')
            % title(i) 
            end
        end
    end
end
sprintf('overlapping filaments segemented = %d / %d',(length(s)- var),segmentcount )
save(strcat(resultspath,sprintf('frame%04d.mat',frame)),'s')
end
toc


%% load "s.mat" data, save csv of end points, centroids, lengths for all filaments

% columns are: Centroid X, Centroid Y, End 1 X, End 1 Y, End 2 X, End 2 Y, Length
frame = 0;
im = imread( strcat(impath,sprintf('Fused_Red_24-02-12_080-083%04d.tif',frame)),'tif' );
%%
for frame = 0
    load(strcat(resultspath,sprintf('frame%04d.mat',frame)));
    thisframedata = zeros(length(s),7);
    extras = zeros(0,7);
    
    for i = 1:length(s)
        filament = zeros(size(im));
        filament(s(i).PixelIdxList) =1;
        %[centrey,centrex] = find(bwmorph(filament,'shrink',Inf));
        [endsy,endsx] = find(bwmorph(filament,'endpoints'));

        % if isscalar(centrey) 
        %     s(i).Centroid = [centrex,centrey];
        % end
        if length(endsy)>1
        thisframedata(i,:) = [s(i).Centroid(1),s(i).Centroid(2),endsx(1),endsy(1),endsx(2),endsy(2),s(i).Length];
        end
        if length(endsy)>2
            for j = 3:length(endsy)
                extras = [extras ; 0 0 endsx(j) endsy(j) 0 0 0];
            end
        end
    end
    writematrix([thisframedata; extras],strcat(resultspath,sprintf('frame%04d.csv',frame)))
    
end