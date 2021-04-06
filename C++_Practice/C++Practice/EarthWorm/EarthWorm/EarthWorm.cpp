// EarthWorm.cpp : 애플리케이션에 대한 진입점을 정의합니다.
//

#include "framework.h"
#include "EarthWorm.h"
#include <math.h>
#include <time.h>
#include <stdlib.h>

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
    LoadStringW(hInstance, IDC_EARTHWORM, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // 애플리케이션 초기화를 수행합니다:
    if (!InitInstance (hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_EARTHWORM));

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
    wcex.hIcon          = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_EARTHWORM));
    wcex.hCursor        = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
    wcex.lpszMenuName   = MAKEINTRESOURCEW(IDC_EARTHWORM);
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
      CW_USEDEFAULT, 0, 500, 500, nullptr, nullptr, hInstance, nullptr);

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
BOOL CrashRecHead(RECT rec1, RECT rec2, int key) // 지렁이 머리, 머리바로 뒤 몸통 충돌시 알림 함수
{
    switch (key)
    {
        case VK_LEFT:
        {
            if (rec1.left == rec2.right) { return TRUE; }
            break;
        }
        case VK_RIGHT:
        {
            if (rec1.right >= rec2.left) { return TRUE; }
            break;
        }
        case VK_UP:
        {
            if (rec1.top <= rec2.bottom) { return TRUE; }
            break;
        }
        case VK_DOWN:
        {
            if (rec1.bottom >= rec2.top) { return TRUE; }
            break;
        }
        default:
        {
            return FALSE;
            break;
        }
    }
};
BOOL CrashRecBody(RECT rec1, RECT rec2) // 지렁이 머리, 몸통 충돌시 알림 함수
{
    if (rec1.left == rec2.right) { return TRUE; }
    else if (rec1.right == rec2.left) { return TRUE; }
    else if (rec1.top == rec2.bottom) { return TRUE; }
    else if (rec1.bottom == rec2.top) { return TRUE; }
    else { return FALSE; }
}
BOOL EatFood(RECT rec1, RECT rec2) // 먹이를 먹었는지 판단할 함수
{
    if (rec1.left == rec2.right)
    {
        if (rec1.top >= rec2.top)
        {
            if (rec1.top - rec2.top <= 10) { return TRUE; }
            else { return FALSE; }
        }
        else if (rec1.top < rec2.top)
        {
            if (rec2.top - rec1.top <= 10) { return TRUE; }
            else { return FALSE; }
        }
        else
        {
            return FALSE;
        }
    }
    else if (rec1.right == rec2.left)
    {
        if (rec1.top >= rec2.top)
        {
            if (rec1.top - rec2.top <= 10) { return TRUE; }
            else { return FALSE; }
        }
        else if (rec1.top < rec2.top)
        {
            if (rec2.top - rec1.top <= 10) { return TRUE; }
            else { return FALSE; }
        }
        else
        {
            return FALSE;
        }
    }
    else if (rec1.bottom == rec2.top)
    {
        if (rec1.right >= rec2.right)
        {
            if (rec1.right - rec2.right <= 10) { return TRUE; }
            else { return FALSE; }
        }
        else if (rec1.right < rec2.right)
        {
            if (rec2.right - rec1.right <= 10) { return TRUE; }
            else { return FALSE; }
        }
        else
        {
            return FALSE;
        }
    }
    else if (rec1.top == rec2.bottom)
    {
        if (rec1.right >= rec2.right)
        {
            if (rec1.right - rec2.right <= 10) { return TRUE; }
            else { return FALSE; }
        }
        else if (rec1.right < rec2.right)
        {
            if (rec2.right - rec1.right <= 10) { return TRUE; }
            else { return FALSE; }
        }
        else
        {
            return FALSE;
        }
    }
    else { return FALSE; }
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    srand((unsigned int)time(NULL));
    static BOOL flag;
    static int nParam;
    
    static RECT Windowrt;                        // 윈도우 크기
    static RECT Worm[50];                        // 지렁이
    static int WormIndex;                        // 지렁이 Index
    static int x, y;
    
    static RECT Food[40];
    static RECT food;
    static int FoodIndex;
    static int count;
    static int up,down,left,right;
    static int up1, down1, left1, right1;

    static int mx, my;

    HBRUSH hbrush, holdbrush;   // 지렁이 색
    HPEN hpen, holdpen;
    SetTimer(hWnd, 1, 10, NULL);
   
    switch (message)
    {
    case WM_CREATE:
        GetClientRect(hWnd, &Windowrt);

        x = 250;
        y = 250;
        WormIndex = 1;

        FoodIndex = 40;
        for (int i = 0; i < FoodIndex; i++)
        {
            mx = rand() % Windowrt.right;
            my = rand() % Windowrt.bottom;

            Food[i].top = my - 5;
            Food[i].left = mx - 5;
            Food[i].bottom = my + 5;
            Food[i].right = mx + 5;
        }
        up = down = left = right = 0;
        up1 = down1 = left1 = right1 = 0;
        count = 0;

        flag = false;

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

            /****************************지렁이 본인*******************************/
            Worm[0].left = x - 5;
            Worm[0].top = y - 5;
            Worm[0].right = x + 5;
            Worm[0].bottom = y + 5;

            hbrush = CreateSolidBrush(RGB(255, 228, 0));
            holdbrush = (HBRUSH)SelectObject(hdc, hbrush);
            // hpen = CreatePen(1, 1, RGB(255, 228, 0));
            // holdpen = (HPEN)SelectObject(hdc, hpen);
            for (int i = 0; i < WormIndex; i++)
            {
                Rectangle(hdc, Worm[i].left, Worm[i].top, Worm[i].right, Worm[i].bottom);
            }
            SelectObject(hdc, holdbrush);
            DeleteObject(hbrush);

            /****************************지렁이 먹이*******************************/
            hbrush = CreateSolidBrush(RGB(188, 229, 92));
            holdbrush = (HBRUSH)SelectObject(hdc, hbrush);
            for (int f = 0; f < FoodIndex; f++)
            {
                Rectangle(hdc, Food[f].left, Food[f].top, Food[f].right, Food[f].bottom);
            }
            SelectObject(hdc, holdbrush);
            DeleteObject(hbrush);
            // SelectObject(hdc, holdpen);
            // DeleteObject(hpen);

            // if (WormIndex > 1)
            // {
            //     if (CrashRecHead(Worm[0], Worm[1], wParam)) // 충돌 여부
            //     {
            //         // MessageBox(hWnd, _T("충돌이냐"), _T("아니 뭐여 이건"), MB_OK | MB_ICONEXCLAMATION);
            //         KillTimer(hWnd, 1);
            //     }
            // }
            /*if(WormIndex > 2)
            {
                for (int i = WormIndex; i > 2; i--)
                {
                    if (CrashRecBody(Worm[0], Worm[i-1]))
                    {
                        // MessageBox(hWnd, _T("충돌이냐"), _T("아니 뭐여 이건"), MB_OK | MB_ICONEXCLAMATION);
                        KillTimer(hWnd, 1);
                    }
                }
            }*/
            
            EndPaint(hWnd, &ps);
        }
        break;
    case WM_KEYDOWN:
        if (FoodIndex == 0)
        {
            KillTimer(hWnd, 1);
            MessageBox(hWnd, _T("게임 끝났다 이말이야"), _T("끝"), MB_OK | MB_ICONEXCLAMATION);
        }
        else
        {
            flag = true;
            nParam = wParam;
        }
        break;
    case WM_KEYUP:
        if (FoodIndex == 0)
        {
            KillTimer(hWnd, 1);
        }
        else
        {
            flag = false;
        }
        break;
    case WM_TIMER:
        
        if (nParam == VK_UP)
        {
            y -= 1;
            if (count != 10) { count++; }
            else
            {
                up = Worm[0].top;
                down = Worm[0].bottom;
                left = Worm[0].left;
                right = Worm[0].right;

                count = 0; 
            }
            if (y - 5 < Windowrt.top) y += 1; 
        }
        else if (nParam == VK_DOWN) 
        {
            y += 1; 
            if (count != 10) { count++; }
            else
            {
                up = Worm[0].top;
                down = Worm[0].bottom;
                left = Worm[0].left;
                right = Worm[0].right;

                count = 0;
            }
            if (y + 5 > Windowrt.bottom) y -= 1; 
        }
        else if (nParam == VK_LEFT) 
        {
            x -= 1; 
            if (count != 10) { count++; }
            else
            {
                up = Worm[0].top;
                down = Worm[0].bottom;
                left = Worm[0].left;
                right = Worm[0].right;

                count = 0;
            }
            if (x - 5 < Windowrt.left) x += 1; 
        }
        else if (nParam == VK_RIGHT) 
        {
            x += 1; 
            if (count != 10) { count++; }
            else
            {
                up = Worm[0].top;
                down = Worm[0].bottom;
                left = Worm[0].left;
                right = Worm[0].right;

                count = 0;
            }
            if (x + 5 > Windowrt.right) x -= 1; 
        }

        if (count == 10)
        {
            for (int i = 1; i < WormIndex; i++) // 꼬리가 머리를 따라가게
            {
                left1 = Worm[i].left;
                up1 = Worm[i].top;
                right1 = Worm[i].right;
                down1 = Worm[i].bottom;

                Worm[i].left = left;
                Worm[i].top = up;
                Worm[i].right = right;
                Worm[i].bottom = down;

                left = left1;
                up = up1;
                right = right1;
                down = down1;
            }
        }

        
        /***********************먹이를 먹었는지 판단************************/
        for (int f = 0; f < FoodIndex; f++)
        {
            if (EatFood(Worm[0], Food[f]))
            {
                WormIndex++; // WormIndex 초기값 1
                
                if (nParam == VK_UP)
                {
                    Worm[WormIndex - 1].top = Worm[WormIndex - 2].bottom;
                    Worm[WormIndex - 1].bottom = Worm[WormIndex - 2].bottom + 10;
                    Worm[WormIndex - 1].left = Worm[WormIndex - 2].left;
                    Worm[WormIndex - 1].right = Worm[WormIndex - 2].right;
                }
                else if (nParam == VK_DOWN) 
                {
                    Worm[WormIndex - 1].top = Worm[WormIndex - 2].top-10;
                    Worm[WormIndex - 1].bottom = Worm[WormIndex - 2].top;
                    Worm[WormIndex - 1].left = Worm[WormIndex - 2].left;
                    Worm[WormIndex - 1].right = Worm[WormIndex - 2].right;
                }
                else if (nParam == VK_LEFT) 
                {
                    Worm[WormIndex - 1].top = Worm[WormIndex - 2].top;
                    Worm[WormIndex - 1].bottom = Worm[WormIndex - 2].bottom;
                    Worm[WormIndex - 1].left = Worm[WormIndex - 2].right;
                    Worm[WormIndex - 1].right = Worm[WormIndex - 2].right+10;
                }
                else if (nParam == VK_RIGHT) 
                {
                    Worm[WormIndex - 1].top = Worm[WormIndex - 2].top;
                    Worm[WormIndex - 1].bottom = Worm[WormIndex - 2].bottom;
                    Worm[WormIndex - 1].left = Worm[WormIndex - 2].left-10;
                    Worm[WormIndex - 1].right = Worm[WormIndex - 2].left;
                }
                
                for (int i = f; i < FoodIndex; i++)     // 먹은 음식 배열 정리
                {
                    Food[i] = Food[i + 1];
                }
                FoodIndex--;
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
