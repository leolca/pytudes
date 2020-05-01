/*
 *  Copyright (c) 2020 Leonardo Araujo (leolca@gmail.com).
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>
 * 
*/
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#define GROUPS 7
const char * group[GROUPS] =
        { "AEIOUHWY", "BFPV", "CGKJQSXZ", "DT", "L", "MN", "R" };
/* to implement the NARA variation use
#define GROUPS 8
const char * group[GROUPS] =
        { "AEIOUY", "BFPV", "CGKJQSXZ", "DT", "L", "MN", "R", "HW" };
#define HW '7'
*/
const char * digit = { "0123456789" };

char soundex(char c)
{
   char r = '?';
   c = toupper(c);
   for(int k=0; k<GROUPS; k++) 
   {
      if (strchr(group[k], c) != (char *)NULL)
      {
        r = digit[k]; break;
      }
   }
   return r;
}

int main(int argc, char *argv[]) 
{
   FILE *stream = stdin;
   char c, sc = 0, previous = 0; 
   int i=0;
   while( (c = fgetc(stream)) != EOF ) 
   {
     if ( !isspace(c) ) 
     {
       if (i == 0) { printf("%c",toupper(c)); previous = soundex(c); i++; }
       else 
       {
	  sc = soundex(c);
	  if (sc != previous && sc > 48 && i < 4)
	  {
	    printf("%c", sc); i++;
	  }
	  previous = sc;
       }
     }
     else
     {
	for(;i<4;i++) printf("0");
	i=0; previous = 0;
        printf("%c",c);
     }
   }
}
