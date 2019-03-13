import csv
import numpy as np
x_train = []
y_train = []
test = []

mean_male_corrected_height,mean_male_corrected_weight,mean_male_corrected_age,mean_female_corrected_height,mean_female_corrected_weight,mean_female_corrected_age,covariance_height = [],[],[],[],[],[],[]
def data_clean(x_train, y_train, test_data):
    with open("akk.csv", encoding='utf-8-sig') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        data = [data for data in rows]

    for x in range(len(data)):
        ins = []
        for y in range(3):
            ins.append(float(data[x][y]))
        if (data[x][y + 1] == "M"):
            y_train.append(float(1))
        else:
            np.array(y_train.append(float(0)))
        x_train.append(ins)

    #print(np.array(x_train))
    #print(np.array(y_train))
    x = np.array(x_train)
    y = np.array(y_train)
    #print("y in data clean", y)
    # x_train = x
    # y_train = y

    with open("test.csv", encoding='utf-8-sig') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        dota = [dota for dota in rows]

    for m in range(len(dota)):
        ins = []
        for l in range(3):
            ins.append(float(dota[m][l]))

        test_data.append(ins)
    test = np.array(test_data)
    # print(test_data)
    print(x)
    print(y)
    print(len(x))
    return x, y, test

def Priors():
    male_count, female_count = 0, 0
    for i in y_train:
        if i == 1:
            male_count += 1

        if i == 0.0:
            female_count += 1


    pb_male = male_count / len(y_train)


    pb_female = female_count / len(y_train)

    print(" the prior probability of male is",pb_male)
    print(" the prior probability of female is",pb_female)
    tot_count = [male_count,female_count,pb_male,pb_female]
    return tot_count

def Calculate_mean():
    tot_count = Priors()
    mean_female_height = mean_male_height = mean_female_weight = mean_male_weight = mean_female_age = mean_male_age = 0
    #print("in calmen y_train[i]", y_train)
    #print("in cal men x_train[i]",x_train)
    #print("len",len(x_train))
    sum_male = sum1_male = sum2_male = 0
    sum_female = sum1_female = sum2_female = 0
    for i in range(len(x_train)):

        #print("now y_train is ",y_train[i])
        if(y_train[i] == 1.0):
            #print("comes in male")
            sum_male +=  x_train[i][0]

            sum1_male += x_train[i][1]
            sum2_male += x_train[i][2]
        if(y_train[i] == 0.0):
            #print("comes in female")
            sum_female += x_train[i][0]

            sum1_female += x_train[i][1]
            sum2_female += x_train[i][2]




    mean_male_height += sum_male / tot_count[0]
    mean_male_weight += sum1_male / tot_count[0]
    mean_male_age += sum2_male / tot_count[0]

    mean_female_height += sum_female / tot_count[1]
    mean_female_weight += sum1_female / tot_count[1]
    mean_female_age += sum2_female / tot_count[1]

    mean_male = [mean_male_height, mean_male_weight, mean_male_age]
    mean_female = [mean_female_height, mean_female_weight, mean_female_age]
    '''
        if (y_train[i] == 0):
            sum_female += x_train[i][0]
            sum1_female += x_train[i][1]
            sum2_female += x_train[i][2]



        

        
    '''
    print("cal mean_male",mean_male)
    print("cal mean_female",mean_female)
    return mean_male,mean_female,tot_count

def Mean_corrected_data(x,y):

    male_data,female_data = [],[]
    mean_male,mean_female,tot_count = Calculate_mean()
    #tot_count = Priors()

    n,m = tot_count[0],tot_count[1]
    print("count of male is",n)
    print("count of female is ",m)
    #mean_corrected_male, mean_corrected_female = np.zeros([3,int(n)]),np.zeros([3,int(m)])
    mean_corrected_male,mean_corrected_female = [],[]
    #print("x here s",x)

    #print("len of x is",len(x))
    mean_height,mean_weight,mean_age = 0,0,0
    for i in range(len(x)):
        if(y[i] == 1):
            male_data.append(x[i])
        else:
            female_data.append(x[i])
    print("male_data",male_data)
    print("female_data",female_data)

    #mean_male,mean_female,tot_count = Calculate_mean()
    tot_count  = Priors()
    #mean_male = np.mean(male_data)
    #mean_female = np.mean(female_data)
    print("mean_male",mean_male)
    print("mean_female",mean_female)
    sum,sum1,sum2 = 0,0,0
    for i in range(len(x)):
        sum += x[i][0]
        sum1 += x[i][1]
        sum2 += x[i][2]

    sum = sum/len(x)
    sum1 = sum1/len(x)
    sum2 = sum2/len(x)
    global_mean = [sum,sum2,sum2]
    print("global_mean is ",global_mean)

    for i in range(len(male_data)):

        mean_corrected_male.append([male_data[i][0] - global_mean[0],male_data[i][1] - global_mean[1],male_data[i][2] - global_mean[2]])

    for i in range(len(female_data)):
        mean_corrected_female.append([female_data[i][0] - global_mean[0], female_data[i][1] - global_mean[1], female_data[i][2] - global_mean[2]])


    print("mean_corrected_male is",mean_corrected_male)
    print("mean_corrected_female is", mean_corrected_female)
    covariance_male = (np.dot(np.array(mean_corrected_male).T,mean_corrected_male))/n
    covariance_female = (np.dot(np.array(mean_corrected_female).T, mean_corrected_female)) / m

    covariance_male = np.array(covariance_male).tolist()
    covariance_female = np.array(covariance_female).tolist()
    print("covariance_male", covariance_male)
    print("covariance_female",covariance_female)
    # pooled within group covariance matrix
    pooled_within_group_covariance = np.zeros([3,3])
    #print("length",len(covariance_male))
    #print("count.ct",tot_count[3])
    #print("cov male 0,2",covariance_female[0][0])
    for i in range(len(covariance_male)):
        #pooled_within_group_covariance
        #print("i is",i)
        #print("covariance_male[i][0]",covariance_male[i][1])
        #print("covariance_female[i][0]",covariance_female[i][1])
        #print("tot_count[4]",tot_count[4])
        #a1 = n * covariance_male[i][0] + tot_count[4] * covariance_female[i][0]
        pooled_within_group_covariance[i][0] = tot_count[2] * covariance_male[i][0] + tot_count[3] * covariance_female[i][0]
        pooled_within_group_covariance[i][1] = tot_count[2] * covariance_male[i][1] + tot_count[3] * covariance_female[i][1]
        pooled_within_group_covariance[i][2] = tot_count[2] * covariance_male[i][2] + tot_count[3] * covariance_female[i][2]
        #pooled_within_group_covariance.append(([tot_count[2] * covariance_male[i][0] + tot_count[3] * covariance_female[i][0],  tot_count[2] * covariance_male[i][1] + tot_count[3] * covariance_female[i][1], tot_count[2] * covariance_male[i][2] + tot_count[3] * covariance_female[i][2]]))
    pooled_inverse = np.linalg.inv(np.array(pooled_within_group_covariance))
    print("pooled within group covariance is ",pooled_within_group_covariance)
    print("pooled_inverse is ",pooled_inverse)
    discriminant_male,discriminant_female= [],[]

    #part_one =  np.matmul(np.matmul(np.array(mean_male),pooled_inverse),np.array(mean_corrected_male).T)
    #part_two = 0.5 * np.matmul((np.matmul(np.array(mean_male),pooled_inverse)),np.array(mean_male).T ) + np.log(tot_count[2])
    part_one = np.matmul(np.matmul(np.array(mean_male), pooled_inverse), np.array(x_train).T)
    part_two = 0.5 * np.matmul((np.matmul(np.array(mean_male), pooled_inverse)), np.array(mean_male).T) + np.log(tot_count[2])
    #print("part_one",part_one)
    #print("print_two",part_two)
    discriminant_male = part_one - part_two
    part_three = np.matmul(np.matmul(np.array(mean_female), pooled_inverse), np.array(x_train).T)
    part_four = 0.5 * np.matmul((np.matmul(np.array(mean_female), pooled_inverse)), np.array(mean_female).T) + np.log(tot_count[3])
    print("mean_male",mean_male)
    print("mean_female",mean_female)
    #print("part_three", part_three)
    #print("print_four", part_four)
    discriminant_female = part_three - part_four
    print("discriminant_male",discriminant_male)
    print("discriminant_female",discriminant_female)
    print("x_train is ",x_train)
    print("y_train is,",y_train)
    predict = []
    for i in range(len(discriminant_male)):
        if(discriminant_male[i] >=discriminant_female[i]):
            predict.append('M')
        else:
            predict.append('W')

    print("The LDA prediction for the given data set is ",predict)












x_train = []
y_train = []
test_data = []
#transpose([[1,2,3],[4,5,6]])
data_clean(x_train,y_train,test_data)

Mean_corrected_data(x_train,y_train)
#covariance(mean_male_corrected_height,mean_male_corrected_weight,mean_male_corrected_age,mean_male_corrected_height,mean_female_corrected_weight,mean_female_corrected_age)
