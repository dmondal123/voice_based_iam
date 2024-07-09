package com.griddynamics.capstone_project.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
@RequestMapping("/api")
public class DetectionController {

    @Autowired
    private DetectionController detectionService;

    @PostMapping("/detect-from-log") // fast api endpoint
    public ResponseEntity<?> processLogFile(@RequestParam("file") MultipartFile file) {
        String result = String.valueOf(detectionService.processLogFile(file));
        return ResponseEntity.ok(result);
    }
}
