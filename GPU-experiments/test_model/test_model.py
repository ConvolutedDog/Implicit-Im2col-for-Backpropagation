import argparse
import csv
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-Implicit")
    parser.add_argument("-model", type=str, help="model config file name")
    parser.add_argument("-batchsize", type=int, help="batch size")
    args = parser.parse_args()
    
    config_dir = "./config/"
    implicit_bin = " ../build/GPUImplicit "
    
    model = args.model
    batchsize = args.batchsize
    
    result_dir = "./result_test/"
    if os.path.exists(result_dir):
        pass
    else:
        os.system("mkdir " + result_dir)
    if os.path.exists(result_dir+model+'.txt'):
        os.system("rm "+result_dir+model+'.txt\n')
    
    
    result =  model + ".txt"
    result_file = result_dir + result
    
    config = open(config_dir+"Sparsity_"+model+".cfg", "r")
    config_readlines = config.readlines()
    config.close()
    
    for line in config_readlines:
        if 'convolution layer' in line:
            items = line.replace("\n", "").split(',')
            if len(items) == 11:
                _, Hi, Wi, Ho, Wo, C, D, Kh, Kw, S, P = items
            elif len(items) == 10:
                P = 0
                _, Hi, Wi, Ho, Wo, C, D, Kh, Kw, S = items # append padding=0
            elif len(items) == 12:
                _, __, Hi, Wi, Ho, Wo, C, D, Kh, Kw, S, P = items # append padding=0
                
            Hi, Wi, Ho, Wo, C, D, Kh, Kw, S, P = int(Hi), int(Wi), int(Ho), \
                int(Wo), int(C), int(D), int(Kh), int(Kw), int(S), int(P)
            
            if (Hi+2*P-Kh)%S != 0:
                Hi = Hi + S - (Hi+2*P-Kh)%S
            if (Wi+2*P-Kw)%S != 0:
                Wi = Wi + S - (Wi+2*P-Kw)%S
            
            cmd_print_info = 'echo Param Info: >> '+result_file+'\n'
            cmd_print_param = 'echo Batchsize='+str(batchsize)+'/ Hi='+str(Hi)+'/ Wi='+str(Wi)+'/ Ho='+str(Ho)+'/ Wo='+str(Wo)+'/ C='+str(C)+'/ D='+\
                             str(D)+'/ Kh='+str(Kh)+'/ Kw='+str(Kw)+'/ S='+str(S)+'/ P='+str(P)+'/ >> '+result_file+'\n'
            cmd_preprocess = implicit_bin+' preprocess {} {} {} {} {} {} {} {} {} {} >> '.format(\
                             batchsize, C, Hi, Wi, D, Kh, Kw, P, S, 1)+result_file+'\n'
            cmd_nopreprocess = implicit_bin+' nopreprocess {} {} {} {} {} {} {} {} {} {} >> '.format(\
                             batchsize, C, Hi, Wi, D, Kh, Kw, P, S, 1)+result_file+'\n'
            
            if S > 1:
                os.system(cmd_print_info)
                os.system(cmd_print_param)
                print(cmd_preprocess)
                os.system(cmd_preprocess)
            
                print(cmd_nopreprocess)
                os.system(cmd_nopreprocess)

    print()
    
    f_result = open(result_file, "r")
    results = f_result.readlines()
    
    for i in range(len(results)):
        line = results[i]
        line = line.replace("\x1b[31m", "").replace("\x1b[0m", "").replace("\x1b[32m", "")
        results[i] = line
    
    print(len(results))
    print(results)
    
    paras_result_index = []
    for i in range(len(results)):
        if 'Param Info' in results[i]:
            paras_result_index.append(i)
    
    print(paras_result_index)
    
    paras_result = []
    print('======================')
    for i in range(len(paras_result_index)):
        if i == len(paras_result_index)-1:
            #print(results[paras_result_index[i]:])
            paras_result.append(results[paras_result_index[i]:])
        else:
            paras_result.append(results[paras_result_index[i]:paras_result_index[i+1]])
    
    print(len(paras_result))
    print('======================')
    
    f_write_log = open('./log_test/'+model + '_batch=' + str(batchsize) + ".txt", "w")
    for para_result in paras_result:
        for line in para_result:
            if 'Param Info' in line:
                f_write_log.writelines('===========\n')
                f_write_log.writelines('Batchsize/Hi/Wi/Ho/Wo/C/D/Kh/Kw/S/P\n')
            if 'Batchsize' in line:
                spline = line.split('/')
                Batchsize, Hi, Wi, Ho, Wo, C, D, Kh, Kw, S, P = spline[0].split('=')[1], spline[1].split('=')[1], \
                    spline[2].split('=')[1], spline[3].split('=')[1], spline[4].split('=')[1], spline[5].split('=')[1], \
                    spline[6].split('=')[1], spline[7].split('=')[1], spline[8].split('=')[1], spline[9].split('=')[1], \
                    spline[10].split('=')[1]
                f_write_log.writelines(Batchsize+'/'+Hi+'/'+Wi+'/'+Ho+'/'+Wo+'/'+C+'/'+D+'/'+Kh+'/'+Kw+'/'+S+'/'+P+'\n')
            if 'PRE Forward Padding time' in line:
                f_write_log.writelines('PRE Forward Padding time/PRE Forward Im2col time/PRE Forward MMA time/PRE Forward workload/PRE Forward Total time/PRE Forward Total gflops/PRE Forward MKN/')
                f_write_log.writelines('PRE Loss Padding time/PRE Loss Im2col time/PRE Loss MMA time/PRE Loss workload/PRE Loss Total time/PRE Loss Total gflops/PRE Loss MKN/')
                f_write_log.writelines('PRE Gradient LossNextLayer Padding time/PRE Gradient Inputfeaturemap Padding time/PRE Gradient Inputfeaturemap Im2col time/PRE Gradient MMA time/PRE Gradient workload/PRE Gradient Total time/PRE Gradient Total gflops/PRE Gradient MKN/')
                f_write_log.writelines('PRENO Forward Im2col time/PRENO Forward MMA time/PRENO Forward workload/PRENO Forward Total time/PRENO Forward Total gflops/PRENO Foreard MKN/')
                f_write_log.writelines('PRENO Loss Im2col time/PRENO Loss MMA time/PRENO Loss workload/PRENO Loss Total time/PRENO Loss Total gflops/PRENO Loss MKN/')
                f_write_log.writelines('PRENO Gradient LossNextLayer Padding time/PRENO Gradient Inputfeaturemap Im2col time/PRENO Gradient MMA time/PRENO Gradient workload/PRENO Gradient Total time/PRENO Gradient Total gflops/PRENO Gradient MKN\n')
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRE Forward Im2col time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRE Forward MMA time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRE Forward workload' in line:
                f_write_log.writelines(line.split(':')[1].split('.')[0]+'/')
            if 'PRE Forward Total time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRE Forward Total gflops' in line:
                f_write_log.writelines(line.split(':')[1].split('gflops')[0]+'/')
            if 'PRE Forward M = ' in line:
                f_write_log.writelines(line.split(',')[0].split('=')[1]+',')
                f_write_log.writelines(line.split(',')[1].split('=')[1]+',')
                f_write_log.writelines(line.split(',')[2].split('=')[1].split('.')[0]+'/')
            
            if 'PRE Loss Padding time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRE Loss Im2col time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRE Loss MMA time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRE Loss workload' in line:
                f_write_log.writelines(line.split(':')[1].split('.')[0]+'/')
            if 'PRE Loss Total time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRE Loss Total gflops' in line:
                f_write_log.writelines(line.split(':')[1].split('gflops')[0]+'/')
            if 'PRE Loss M = ' in line:
                f_write_log.writelines(line.split(',')[0].split('=')[1]+',')
                f_write_log.writelines(line.split(',')[1].split('=')[1]+',')
                f_write_log.writelines(line.split(',')[2].split('=')[1].split('.')[0]+'/')
        
            if 'PRE Gradient LossNextLayer Padding time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRE Gradient Inputfeaturemap Padding time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRE Gradient Im2col time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRE Gradient MMA time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRE Gradient workload' in line:
                f_write_log.writelines(line.split(':')[1].split('.')[0]+'/')
            if 'PRE Gradient Total time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRE Gradient Total gflops' in line:
                f_write_log.writelines(line.split(':')[1].split('gflops')[0]+'/')
            if 'PRE Gradient M = ' in line:
                f_write_log.writelines(line.split(',')[0].split('=')[1]+',')
                f_write_log.writelines(line.split(',')[1].split('=')[1]+',')
                f_write_log.writelines(line.split(',')[2].split('=')[1].split('.')[0]+'/')
            
            
            if 'PRENO Forward Im2col time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRENO Forward MMA time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRENO Forward workload' in line:
                f_write_log.writelines(line.split(':')[1].split('.')[0]+'/')
            if 'PRENO Forward Total time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRENO Forward Total gflops' in line:
                f_write_log.writelines(line.split(':')[1].split('gflops')[0]+'/')
            if 'PRENO Forward M = ' in line:
                f_write_log.writelines(line.split(',')[0].split('=')[1]+',')
                f_write_log.writelines(line.split(',')[1].split('=')[1]+',')
                f_write_log.writelines(line.split(',')[2].split('=')[1].split('.')[0]+'/')
            
            if 'PRENO Loss Im2col time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRENO Loss MMA time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRENO Loss workload' in line:
                f_write_log.writelines(line.split(':')[1].split('.')[0]+'/')
            if 'PRENO Loss Total time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRENO Loss Total gflops' in line:
                f_write_log.writelines(line.split(':')[1].split('gflops')[0]+'/')
            if 'PRENO Loss M = ' in line:
                f_write_log.writelines(line.split(',')[0].split('=')[1]+',')
                f_write_log.writelines(line.split(',')[1].split('=')[1]+',')
                f_write_log.writelines(line.split(',')[2].split('=')[1].split('.')[0]+'/')
        
            if 'PRENO Gradient LossNextLayer Padding time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRENO Gradient Im2col time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRENO Gradient MMA time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRENO Gradient workload' in line:
                f_write_log.writelines(line.split(':')[1].split('.')[0]+'/')
            if 'PRENO Gradient Total time' in line:
                f_write_log.writelines(line.split(':')[1].split('ms')[0]+'/')
            if 'PRENO Gradient Total gflops' in line:
                f_write_log.writelines(line.split(':')[1].split('gflops')[0]+'/')
            if 'PRENO Gradient M = ' in line:
                f_write_log.writelines(line.split(',')[0].split('=')[1]+',')
                f_write_log.writelines(line.split(',')[1].split('=')[1]+',')
                f_write_log.writelines(line.split(',')[2].split('=')[1].split('.')[0]+'\n')
            
        
        
        
        
        
    f_write_log.close()
    f_result.close()
    
    
