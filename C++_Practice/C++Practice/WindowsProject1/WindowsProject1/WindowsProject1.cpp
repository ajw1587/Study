// WindowsProject1.cpp : 애플리케이션에 대한 진입점을 정의합니다.
//

#include "framework.h"
#include "WindowsProject1.h"
#include <math.h>

#define MAX_LOADSTRING 100
#define MAX_LINESTRING 100

// 전역 변수:
HINSTANCE hInst;                                // 현재 인스턴스입니다.
WCHAR szTitle[MAX_LOADSTRING];                  // 제목 표시줄 텍스트입니다.
WCHAR szWindowClass[MAX_LOADSTRING];            // 기본 창 클래스 이름입니다.

// 이 코드 모듈에 포함된 함수의 선언을 전달합니다:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

int DDlength(int left, int top, int right, int bottom)
{
    return sqrt(pow((double)(left-right),5.0f)+ pow((double)(top - bottom), 5.0f));
}

BOOL IsinCircle(int left, int top, int right, int bottom, int size)
{
    return DDlength(left, top, right, bottom) < size ? true : false;
}


int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
                     _In_opt_ HINSTANCE hPrevInstance,
                     _In_ LPWSTR    lpCmdLine,
                     _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    // TODO: 여기에 코드를 입력합니다.

    // 전역 문자열을 초기화합니다.
    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_WINDOWSPROJECT1, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // 애플리케이션 초기화를 수행합니다:
    if (!InitInstance (hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_WINDOWSPROJECT1));

    MSG msg;

    // 기본 메시지 루프입니다:
    while (GetMessage(&msg, nullptr, 0, 0))
    {
        if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    return (int) msg.wParam;
}



//
//  함수: MyRegisterClass()
//
//  용도: 창 클래스를 등록합니다.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEXW wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style          = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc    = WndProc;
    wcex.cbClsExtra     = 0;
    wcex.cbWndExtra     = 0;
    wcex.hInstance      = hInstance;
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_WINDOWSPROJECT1));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_WINDOWSPROJECT1);
    wcex.lpszClassName  = szWindowClass;
    wcex.hIconSm        = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

    return RegisterClassExW(&wcex);
}

//
//   함수: InitInstance(HINSTANCE, int)
//
//   용도: 인스턴스 핸들을 저장하고 주 창을 만듭니다.
//
//   주석:
//
//        이 함수를 통해 인스턴스 핸들을 전역 변수에 저장하고
//        주 프로그램 창을 만든 다음 표시합니다.
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
   hInst = hInstance; // 인스턴스 핸들을 전역 변수에 저장합니다.

   HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW | WS_VSCROLL | WS_HSCROLL,
      CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

   if (!hWnd)
   {
      return FALSE;
   }

   ShowWindow(hWnd, nCmdShow);
   UpdateWindow(hWnd);

   return TRUE;
}

//
//  함수: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  용도: 주 창의 메시지를 처리합니다.
//
//  WM_COMMAND  - 애플리케이션 메뉴를 처리합니다.
//  WM_PAINT    - 주 창을 그립니다.
//  WM_DESTROY  - 종료 메시지를 게시하고 반환합니다.
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    static TCHAR str[MAX_LINESTRING][MAX_LOADSTRING];
    static int index;
    static int cols;
    static SIZE size; // 캐럿 사이즈
    
    /****************************************가상 키보드***********************************************/
    static BOOL uprt;
    static BOOL downrt;
    static BOOL leftrt;
    static BOOL rightrt;

    static RECT rt1;
    rt1.left = 200;
    rt1.top = 200;
    rt1.bottom = 300;
    rt1.right = 300;

    static RECT rt2;
    rt2.left = 200;
    rt2.top = 300;
    rt2.bottom = 400;
    rt2.right = 300;

    static RECT rt3;
    rt3.left = 100;
    rt3.top = 300;
    rt3.bottom = 400;
    rt3.right = 200;

    static RECT rt4;
    rt4.left = 300;
    rt4.top = 300;
    rt4.bottom = 400;
    rt4.right = 400;

    HPEN hpen, holdpen;
    hpen = CreatePen(PS_DOT, 5, RGB(255, 0, 0));
    HBRUSH hbrush, holdbrush;
    HBRUSH cbrush, coldbrush;

    static BOOL flag;
    static int nParam;
    /**************************************************************************************************/
    /****************************************움직이는 도형***********************************************/
    static RECT rt;                    // 화면 크기 도형
    static int x, y;                   // 도형 위치값
    static int mx, my;                 // 변형된 도형 위치값
    static bool Selection;             // 마우스 좌클릭으로 도형 클릭시 커서가 도형 안에 있는지 파악하는 변수
    SetTimer(hWnd, 1, 50, NULL);

    switch (message)
    {
    case WM_COMMAND:
        {
            int wmId = LOWORD(wParam);
            // 메뉴 선택을 구문 분석합니다:
            switch (wmId)
            {
            case IDM_ABOUT:
                DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
                break;
            case IDM_EXIT:
                DestroyWindow(hWnd);
                break;
            default:
                return DefWindowProc(hWnd, message, wParam, lParam);
            }
        }
        break;
    case WM_CREATE:
        flag = FALSE;
        Selection = FALSE;
        GetClientRect(hWnd, &rt);
        x, y = 20;
        mx = my = 0;

        CreateCaret(hWnd, NULL, 2, 16);
        ShowCaret(hWnd);
        index = 0;
        cols = 0;
        break;

    case WM_CHAR:
        
        if (wParam == VK_BACK)
        {
            if (cols == 0 && index == 0) break;
            else if (cols != 0 && index == 0)
            {
                index = _tcslen(str[cols - 1]);
                cols--;
            }
            else
            {
                index--;
                str[cols][index] = NULL;
            }
        }
        else if (wParam == VK_RETURN || index == MAX_LOADSTRING - 1)
        {
            cols++;
            index = 0;
        }
        else
        {
            str[cols][index++] = wParam;
            str[cols][index] = NULL;
        }
        InvalidateRgn(hWnd, NULL /* 전체 출력 */, TRUE);
        break;

    case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);
           
            // TODO: 여기에 hdc를 사용하는 그리기 코드를 추가합니다...

            /****************************************가상 키보드***********************************************/
            
            uprt = Rectangle(hdc, 200, 200, 300, 300);
            DrawText(hdc, _T("↑"), _tcslen(_T("↑")), &rt1, DT_SINGLELINE | DT_CENTER | DT_VCENTER ); //OK
            
            downrt = Rectangle(hdc, 200, 300, 300, 400);
            DrawText(hdc, _T("↓"), _tcslen(_T("↓")), &rt2, DT_SINGLELINE | DT_CENTER | DT_VCENTER);
                       
            leftrt = Rectangle(hdc, 100, 300, 200, 400);
            DrawText(hdc, _T("←"), _tcslen(_T("←")), &rt3, DT_SINGLELINE | DT_CENTER | DT_VCENTER);
                        
            rightrt = Rectangle(hdc, 300, 300, 400, 400);
            DrawText(hdc, _T("→"), _tcslen(_T("→")), &rt4, DT_SINGLELINE | DT_CENTER | DT_VCENTER); //OK

            if (flag == true)
            {
                hbrush = CreateSolidBrush(RGB(255, 0, 0));
                holdbrush = (HBRUSH)SelectObject(hdc, hbrush);
                switch (nParam)
                {
                case VK_UP:
                    Rectangle(hdc, 200, 200, 300, 300);
                    break;
                case VK_DOWN:
                    Rectangle(hdc, 200, 300, 300, 400);
                    break;
                case VK_LEFT:
                    Rectangle(hdc, 100, 300, 200, 400);
                    break;
                case VK_RIGHT:
                    Rectangle(hdc, 300, 300, 400, 400);
                    break;
                }
                SelectObject(hdc, holdbrush);
                DeleteObject(hbrush);
            }
            /**************************************************************************************************/

            /**************************************메모장 기능*****************************************/
            //TextOut(hdc, 150, 150, _T("hello world"), _tcslen(_T("hello world")));                              // 일반 텍스트 위치선정 및 출력
            //DrawText(hdc, _T("Bye World"), _tcslen(_T("Bye World")), &rt, DT_SINGLELINE | DT_LEFT | DT_BOTTOM); // 상자안에 텍스트 출력

            
            for (int i = 0; i <= cols; i++)
            { 
                MoveToEx(hdc, 0, cols * 16+18, NULL);               // 심심해서 밑줄 만들어본거
                LineTo(hdc, 150, cols * 16+18);                     // 심심해서 밑줄 만들어본거
                TextOut(hdc, 0, i * 16, str[i], _tcslen(str[i])); 
            }
            GetTextExtentPoint(hdc, str[cols], _tcslen(str[cols]), &size); // 문자열의 위치값
            SetCaretPos(size.cx, cols * 16);                               // 커서 셋팅
            /**************************************************************************************************/
            /**************************************도형 움직이기*****************************************/
            if (flag == TRUE || Selection == TRUE)
            {
                hbrush = CreateSolidBrush(RGB(0, 0, 255));
                holdbrush = (HBRUSH)SelectObject(hdc, hbrush);

                Ellipse(hdc, x - 20, y - 20, x + 20, y + 20);

                SelectObject(hdc, holdbrush);
                DeleteObject(hbrush);
            }
            else { Ellipse(hdc, x - 20, y - 20, x + 20, y + 20); }

            
            
            /**************************************************************************************************/
            DeleteObject(hpen);
            EndPaint(hWnd, &ps);
        }
        break;
    case WM_SIZE:
        GetClientRect(hWnd, &rt);
        
    case WM_KEYDOWN:
        flag = true;
        nParam = wParam;
        // holdpen = (HPEN)SelectObject(hdc, hpen); hdc 값이 선언 되어있지 않다.

        /**************************************도형 움직이기*****************************************/
        // if (nParam == VK_RIGHT) { x += 5; if (x + 5 > rt.right) { x -= 5; } }
        // else if (nParam == VK_LEFT) { x -= 5; if (x - 5 < rt.left) { x += 5; } }
        // else if (nParam == VK_UP) { y -= 5; if (y - 5 < rt.top) { y += 5; } }
        // else if (nParam == VK_DOWN) { y += 5; if (y + 5 > rt.bottom) { y -= 5; } }

        InvalidateRgn(hWnd, NULL, TRUE);
        break;
    case WM_KEYUP:
        flag = false;
        InvalidateRgn(hWnd, NULL, TRUE);
        break;
    case WM_TIMER:
        /**************************************도형 움직이기*****************************************/
        if (nParam == VK_RIGHT) { x += 5; if (x + 20 > rt.right) { x -= 20; } }
        else if (nParam == VK_LEFT) { x -= 5; if (x - 20 < rt.left) { x += 20; } }
        else if (nParam == VK_UP) { y -= 5; if (y - 20 < rt.top) { y += 20; } }
        else if (nParam == VK_DOWN) { y += 5; if (y + 20 > rt.bottom) { y -= 20; } }
        InvalidateRgn(hWnd, NULL, TRUE);

        break;
    case WM_LBUTTONDOWN:
        KillTimer(hWnd, 1);
        mx = LOWORD(lParam);
        my = HIWORD(lParam);
        if (IsinCircle(x, y, mx, my, 20))
        {
            Selection = TRUE;
            flag = TRUE;
        }
        
        break;
    case WM_MOUSEMOVE:
        if (Selection)
        {
            x = LOWORD(lParam), y = HIWORD(lParam);
            KillTimer(hWnd, 1);
        }
        InvalidateRgn(hWnd, NULL, TRUE);
        break;
    case WM_LBUTTONUP:
        Selection = FALSE;
        flag = FALSE;

        break;
    case WM_DESTROY:
        KillTimer(hWnd, 1);
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// 정보 대화 상자의 메시지 처리기입니다.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(lParam);
    switch (message)
    {
    case WM_INITDIALOG:
        return (INT_PTR)TRUE;

    case WM_COMMAND:
        if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
        {
            EndDialog(hDlg, LOWORD(wParam));
            return (INT_PTR)TRUE;
        }
        break;
    }
    return (INT_PTR)FALSE;
}
