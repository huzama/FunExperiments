#include<Windows.h>

//Entry Point
int CALLBACK WinMain(
	HINSTANCE hInstance,
	HINSTANCE hPrevInstance,
	LPSTR     lpCmdLine,
	int       nShowCmd)
{
	//Declaing x Name
	const auto ClasName = "classText";


	//Creating Sturt For Regisring Class
	WNDCLASSEX WindowsClass = { 0 };
	
	//Setting Sturct Values
	WindowsClass.cbSize = sizeof(WindowsClass);
	WindowsClass.style = CS_OWNDC;
	WindowsClass.lpfnWndProc = DefWindowProc;
	WindowsClass.cbClsExtra = 0;
	WindowsClass.cbWndExtra = 0;
	WindowsClass.hInstance = hInstance;
	WindowsClass.hIcon = NULL;
	WindowsClass.hCursor = NULL;
	WindowsClass.hbrBackground = NULL;
	WindowsClass.lpszMenuName = NULL;
	WindowsClass.hIconSm = NULL;
	WindowsClass.lpszClassName = ClasName;

	//Register class
	RegisterClassEx(&WindowsClass);

	//Create Instance

	auto WindowsInst = CreateWindowEx(0,
		ClasName,
		"Text",
		WS_CAPTION | WS_MINIMIZE | WS_SYSMENU,
		200, 200, 1280, 720,
		NULL, NULL,	hInstance, NULL);


	ShowWindow(WindowsInst, SW_SHOW);

	while (1);
	return 0;
}