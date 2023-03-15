import { Component, OnInit } from '@angular/core';
import { GETrequestService } from '../getrequest.service';
import { MoviesDB } from '../MoviesDBs';

@Component({
  selector: 'search',
  templateUrl: './search-bar.component.html',
  styleUrls: ['./search-bar.component.css']
})
export class SearchBarComponent implements OnInit {

  constructor(private _GetDataService: GETrequestService) { }
 
  public DATA:MoviesDB ;
  public Titles = [];
  public detail;
 
  ngOnInit() {

  }
  onType(event)
  {


    console.clear();
    console.log(event);
    
    this.Titles = [];
    var Name = document.getElementById("myInput").value;
    var Year = document.getElementById("Years").value;
    
    this._GetDataService.GET(Name,Year).subscribe(data=> this.DATA = data);
    
    for(var index in this.DATA.Search)
    { 
            this.Titles[index] = this.DATA.Search[index]; 
    }
    console.clear();

  }

  onSelct(imdb:string)
  {
    console.clear();
    this._GetDataService.OPENPAGE(imdb).subscribe(data=> this.detail = data);  
   
    if(this.detail != undefined)
    {
     // this.Titles = [];
    }
    console.clear();
    
  }
  
}