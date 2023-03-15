#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include<CL\cl.hpp>
#include<fstream>
#include<string>
#include<iostream>

int main()
{
	std::vector<cl::Platform> myPlatforms;
	cl::Platform::get(&myPlatforms);


	{  
		// For printing info about supported devices and Platform(APIs)
		std::cout << "Huzama Industry." << std::endl << std::endl;
		int noOfPlatforms = myPlatforms.size();
		_ASSERT(noOfPlatforms > 0);
		std::cout << "There are " << noOfPlatforms << " OpenCL Supported Platform." << std::endl;

		for (int j = 0; j < noOfPlatforms; j++) 
		{
			std::cout << std::endl << "For Platform " << j << std::endl;
			auto platform = myPlatforms[j];
			std::vector<cl::Device> myDevices;
			platform.getDevices(CL_DEVICE_TYPE_ALL, &myDevices);

			int noOfDevices = myDevices.size();
			_ASSERT(noOfDevices > 0);

			std::cout << "There are " << noOfDevices << " OpenCL Supported Devices." << std::endl << std::endl;

			for (int i = 0; i < noOfDevices; i++)
			{
				std::cout << i + 1 << ". ";
				auto Device = myDevices[i];
				std::cout << Device.getInfo<CL_DEVICE_VENDOR>() << Device.getInfo<CL_DEVICE_VERSION>() << std::endl;
			}
		}
	}


	_ASSERT(myPlatforms.size() > 0);
	auto platform = myPlatforms.front();
	std::vector<cl::Device> myDevices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &myDevices);

	_ASSERT(myDevices.size() > 0);

	std::ifstream myFile("Kernel.cl");
	std::string myString(std::istreambuf_iterator<char>(myFile), (std::istreambuf_iterator<char>()));
	cl::Program::Sources mySource(1, std::make_pair(myString.c_str(), myString.length() + 1));

	cl::Context context(myDevices);
	cl::Program  program(context, mySource);
	program.build("-cl-std=CL1.2");

	return 69;
}