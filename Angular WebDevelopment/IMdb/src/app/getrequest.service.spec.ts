import { TestBed } from '@angular/core/testing';

import { GETrequestService } from './getrequest.service';

describe('GETrequestService', () => {
  beforeEach(() => TestBed.configureTestingModule({}));

  it('should be created', () => {
    const service: GETrequestService = TestBed.get(GETrequestService);
    expect(service).toBeTruthy();
  });
});
