#include <windows.h>
#include <iostream>


int main()
{
	/*LPSTR Dir = (LPSTR)"G:\\file";

	if (!CreateDirectory(Dir, NULL))
		std::cout << "Dir cannot be created, Error: " << GetLastError() << std::endl;
	else
	std::cout << "Dir " << Dir << " succefully Created" << std::endl;


	if (!CopyFile("D:\\file.txt", "G:\\file\\file.txt", 0))
		std::cout << "File cannot be copied, Error: " << GetLastError() << std::endl;
	else
	std::cout << "File Copied" << std::endl;
	*/


	HANDLE file = CreateFile("G:\\Project_Data\\Video.mkv", GENERIC_READ , FILE_SHARE_READ, NULL, OPEN_EXISTING, NULL, NULL);
	if (file == INVALID_HANDLE_VALUE)
	{
		std::cout << GetLastError();
	}

	char Buf[4000] = { 0 };
	DWORD myINT = 1;
	
	unsigned long int i = 0;


	while (myINT)
	{
		ReadFile(file, Buf, 4000, &myINT, NULL);
		i++;
	}
	
	i = i * 4000 + myINT;

	for (int ij = 0; ij < 4000; ij++)
		std::cout << Buf[ij];

	CloseHandle(file);
	return 0;
}