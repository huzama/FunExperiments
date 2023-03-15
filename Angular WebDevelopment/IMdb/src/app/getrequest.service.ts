import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { MoviesDB } from './MoviesDBs'
import { Observable } from 'rxjs';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class GETrequestService {

  private np = new BehaviorSubject<string>("defaut message");
currentmessage= this.np.asObservable();
changemessage(name:string){
  this.np.next(name);
}

  constructor(private http: HttpClient) { }

  GET(NAME, Year):Observable<MoviesDB>
  {
    var SendURL = "http://www.omdbapi.com/?apikey=b4bb3e96&type=movie&s=" + NAME + "&y="+Year;
    return this.http.get<MoviesDB>(SendURL);
  }

  OPENPAGE(imdb:string)
  {
    return this.http.get("http://www.omdbapi.com/?apikey=b4bb3e96&i="+imdb);
  }
}
