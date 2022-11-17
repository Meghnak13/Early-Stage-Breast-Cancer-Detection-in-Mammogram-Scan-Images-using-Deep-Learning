#import data_processing
#import GenerativeAdversarialNetwork
#import cgan
#import ImprovedGAN
import make_chart
#import matplotlib.pyplot as plt

def main():
    # Main Function
    print("main() function loaded")
    #ImprovedGAN.train()
    #print("training completed!")
    #ImprovedGAN.test_model()
    make_chart.create_chart()

    #obj = cgan.CGAN()
    #obj.train(10000, 8, 50)


    #data_processing.test_function()
    #GenerativeAdversarialNetwork.run_GAN()
    #data_processing.show_samples(data_processing.load_DataFrame())
    #GenerativeAdversarialNetwork.run_GAN()



if __name__ == "__main__":
    main()
    