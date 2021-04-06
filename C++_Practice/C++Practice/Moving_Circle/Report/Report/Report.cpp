// Report.cpp : 애플리케이션에 대한 진입점을 정의합니다.
//

#include "framework.h"
#include "Report.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define MAX_LOADSTRING 100

// 전역 변수:
HINSTANCE hInst;                                // 현재 인스턴스입니다.
WCHAR szTitle[MAX_LOADSTRING];                  // 제목 표시줄 텍스트입니다.
WCHAR szWindowClass[MAX_LOADSTRING];            // 기본 창 클래스 이름입니다.

// 이 코드 모듈에 포함된 함수의 선언을 전달합니다:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

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
    LoadStringW(hInstance, IDC_REPORT, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // 애플리케이션 초기화를 수행합니다:
    if (!InitInstance (hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_REPORT));

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
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_REPORT));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_REPORT);
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

   HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
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

int GetVelocity(int i)
{
    return i *= 5 ;
}
double GetLength(int x, int y, int mx, int my)
{
    return sqrt(pow((double)(x - mx), 2.0f) + pow((double)(y - my), 2.0f));
}
BOOL IsinCircle(int x, int y, int mx, int my)
{
    return GetLength(x, y, mx, my) < 46;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    static RECT Windowrt;
    static int v;
    static int x, y;
    static const int size = 20;
    static int mx, my;
    static BOOL identify;
    static int random1;
    int random2;
    SetTimer(hWnd, 1, 10, NULL);
    srand((unsigned int)time(NULL));
    static BOOL Selection;

    HBRUSH hbrush;
    HBRUSH holdbrush;

    switch (message)
    {
    case WM_CREATE:
        
        GetClientRect(hWnd, &Windowrt);
        v = 1;
        x = y = 20;
        mx = my = 0;
        random1 = 0;
        random2 = 0;
        Selection = FALSE;
        identify = FALSE;
        break;
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
    case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hWnd, &ps);
            // TODO: 여기에 hdc를 사용하는 그리기 코드를 추가합니다...

            hbrush = CreateSolidBrush(RGB(0, 0, 150));
            holdbrush = (HBRUSH)SelectObject(hdc, hbrush);

            Ellipse(hdc, x-20, y-20, x + 20, y + 20);
            
            SelectObject(hdc, holdbrush);
            DeleteObject(hbrush);
            EndPaint(hWnd, &ps);
        }
        break;
    case WM_LBUTTONDOWN:
        identify = TRUE;

        mx = LOWORD(lParam);
        my = HIWORD(lParam);
        if (IsinCircle(x, y, mx, my))
        {
            KillTimer(hWnd, 1); // 클릭을 해도 왜 멈추지 않지??
            Selection = TRUE;
        }
        InvalidateRgn(hWnd, NULL, TRUE);
        break;
    case WM_MOUSEMOVE:
        if (Selection)
        {
            x = LOWORD(lParam);
            y = HIWORD(lParam);
            KillTimer(hWnd, 1);
            InvalidateRgn(hWnd, NULL, TRUE);
        }

        break;
    case WM_LBUTTONUP:
        identify = FALSE;
        SetTimer(hWnd, 1, 10, NULL);
        Selection = FALSE;
        InvalidateRgn(hWnd, NULL, TRUE);
        break;

    case WM_TIMER:

        if (!identify)
        {
            // random1 = rand() % 2 + 1;
            if (y + 20 <= Windowrt.bottom)
            {
                y += 5; //GetVelocity(v); // 왜 속도가 빨리지지 않지??
            }
            else // y + 20 > Windowrt.bottom;
            {
                random1 = rand() % 2 + 1;
                switch (random1)
                {
                case 1:
                    random2 = (rand() % (Windowrt.right - x - 20)) / 5;
                    for (int i = 0; i < random2; i++)
                    {
                        if (x + 20 < Windowrt.right)
                        {
                          x += 1;
                        }
                        else { break; }
                    }
                    break;
                case 2:
                    random2 = (rand() % (x - 20)) / 5;
                    for (int j = 0; j < random2; j++)
                    {
                        if (x - 20 > Windowrt.left)
                        {
                            x -= 1;
                        }
                        else { break; }
                    }
                    break;
                }
            }
        }
        InvalidateRgn(hWnd, NULL, TRUE);
        break;

    case WM_SIZE:
        GetClientRect(hWnd, &Windowrt);
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
