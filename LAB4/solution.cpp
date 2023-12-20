//Do not edit the code below (unless you know what you are doing)
#include <math.h>
#include"solution.h"
# define M_PI           3.14159265358979323846
int solution::f_calls = 0;
int solution::g_calls = 0;
int solution::H_calls = 0;

solution::solution(double L)
{
	x = matrix(L);
	//g = NAN;
	//H = NAN;
	g = matrix(2, 1);
	H = matrix(2, 2);
}

solution::solution(const matrix &A)
{
	x = A;
	//g = NAN;
	//H = NAN;
	g = matrix(new double[2]{ 0.,0. }, 2);
	H = matrix(2, 2);
}

solution::solution(double *A, int n)
{
	x = matrix(A, n);
	//g = NAN;
	//H = NAN;
	g = matrix(new double[2]{ 0.,0. }, 2);
	H = matrix(2, 2);
}

void solution::clear_calls()
{
	f_calls = 0;
	g_calls = 0;
	H_calls = 0;
}

ostream &operator<<(ostream &S, const solution &A)
{
	S << "x = " << A.x << endl;
	S << "y = " << A.y << endl;
	S << "f_calls = " << solution::f_calls << endl;
	S << "g_calls = " << solution::g_calls << endl;
	S << "H_calls = " << solution::H_calls << endl;
	return S;
}

//You can edit the following code

void solution::fit_fun(matrix O)
{
	//y =  -cos(0.1* x(0,0)) * exp(-1* pow(0.1* x(0, 0) - 2* M_PI, 2))+ 0.002*pow(0.1* x(0, 0) ,2);
	//y = x(0, 0) * x(0, 0) * x(0, 0) * x(0, 0) + x(0, 0) * x(0, 0) * x(0, 0) - x(0, 0) * x(0, 0) + 2 * x(0, 0) - 6;//check for expansion
	//y = x(0, 0) + (1 / (x(0, 0) * x(0, 0)));//check Fib and  LAgrange
	//y = pow(x(0), 2) + pow(x(1), 2) - cos(2.5 * M_PI * x(0)) - cos(2.5 * M_PI * x(1)) + 2;
	//y = 2.5 * pow((pow(x(0), 2) - x(1)), 2)+ pow((1- x(1)),2);
	//y = pow(x(0), 2) + pow(x(1), 2);
	/*if (g_calls < 100) {
		y = sin(M_PI * sqrt(pow(x(0) / M_PI, 2) + pow(x(1) / M_PI, 2))) / (M_PI * sqrt(pow(x(0) / M_PI, 2) + pow(x(1) / M_PI, 2)));

		if (-x(0) + 1 > 0) {
			y = y + O(0) * pow(-x(0) + 1, 2);
		}
		if (-x(1) + 1 > 0) {
			y = y + O(0) * pow(-x(1) + 1, 2);
		}
		if (sqrt(pow(x(0), 2) + pow(x(1), 2)) - O(1) > 0) {
			y = y + O(0) * pow(sqrt(pow(x(0), 2) + pow(x(1), 2)) - O(1), 2);
		}
	

	}
	else {

		y = sin(M_PI * sqrt(pow(x(0) / M_PI, 2) + pow(x(1) / M_PI, 2))) / (M_PI * sqrt(pow(x(0) / M_PI, 2) + pow(x(1) / M_PI, 2)));
	
	if (-x(0) + 1 > 0) {
		y = 1e10;
	}
	else {
		y = y - (O(0) / (-x(0) + 1));
	}
	if (-x(1) + 1 > 0) {
		y = 1e10;
	}
	else {
		y = y - (O(0) / (-x(1) + 1));
	}
	if (sqrt(pow(x(0), 2) + pow(x(1), 2)) - O(1) > 0) {
		y = 1e10;
	}
	else {
		y = y - (O(0) / (sqrt(pow(x(0), 2) + pow(x(1), 2)) - O(1)));
	}
}*/
	int* n = get_size(O);

	if (n[1] == 1) {
		y = pow(x(0) + x(1) * 2 - 7, 2) + pow(x(0) * 2 + x(1) - 5, 2);
		++f_calls;
	}
	else {
		solution tmp;
		tmp.x = O[0] + x * O[1];
		tmp.fit_fun();
		y = tmp.y;
	}


	++f_calls;

}

void solution::grad(matrix O)
{

	g(0) = 10 * x(0) + 8 * x(1) - 34;
	g(1) = 8 * x(0) + 10 * x(1) - 38;


	++g_calls;
}

void solution::hess(matrix O)
{

	H(0, 0) = 10;
	H(0, 1) = 8;
	H(1, 0) = 8;
	H(1, 1) = 10;

	++H_calls;
}
